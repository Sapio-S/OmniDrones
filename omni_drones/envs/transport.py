import functorch

import omni.isaac.core.utils.torch as torch_utils
import torch
from omni.isaac.core.objects import DynamicCuboid
from tensordict.tensordict import TensorDict, TensorDictBase
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec

import omni_drones.utils.kit as kit_utils
import omni_drones.utils.scene as scene_utils

from omni_drones.envs.isaac_env import AgentSpec, IsaacEnv
from omni_drones.views import RigidPrimView
from omni_drones.utils.torch import cpos, off_diag
from omni_drones.robots.assembly.transportation_group import TransportationGroup
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase


class Transport(IsaacEnv):
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.group.initialize()
        self.payload = self.group.payload_view
        self.payload_target_visual = RigidPrimView(
            "/World/envs/.*/payloadTargetVis",
            reset_xform_properties=False
        )
        self.payload_target_visual.initialize()

        self.init_poses = self.group.get_world_poses(clone=True)
        self.init_velocities = torch.zeros_like(self.group.get_velocities())
        self.init_joint_pos = self.group.get_joint_positions(clone=True)
        self.init_joint_vel = torch.zeros_like(self.group.get_joint_velocities())

        self.init_drone_poses = self.drone.get_world_poses(clone=True)
        self.init_drone_vels = torch.zeros_like(self.drone.get_velocities())

        drone_state_dim = self.drone.state_spec.shape[0]
        observation_spec = CompositeSpec({
            "self": UnboundedContinuousTensorSpec((1, drone_state_dim)).to(self.device),
            "others": UnboundedContinuousTensorSpec((self.drone.n-1, 4)).to(self.device),
            "payload": UnboundedContinuousTensorSpec((1, 19)).to(self.device)
        })

        state_spec = CompositeSpec(
            drones=UnboundedContinuousTensorSpec((self.drone.n, drone_state_dim)).to(self.device),
            payload=UnboundedContinuousTensorSpec((1, 16)).to(self.device)
        )

        self.agent_spec["drone"] = AgentSpec(
            "drone",
            4,
            observation_spec,
            self.drone.action_spec.to(self.device),
            UnboundedContinuousTensorSpec(2).to(self.device),
            state_spec=state_spec
        )

        self.payload_target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.payload_target_heading = torch.zeros(self.num_envs, 3, device=self.device)

    def _design_scene(self):
        cfg = RobotCfg()
        self.drone: MultirotorBase = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)
        self.group = TransportationGroup(drone=self.drone)

        scene_utils.design_scene()

        self.group.spawn(translations=[(0, 0, 2.0)])
        DynamicCuboid(
            "/World/envs/env_0/payloadTargetVis",
            scale=torch.tensor([0.5, 0.5, 0.2]),
            color=torch.tensor([0.8, 0.1, 0.1]),
            size=2.01,
        )
        kit_utils.set_collision_properties(
            "/World/envs/env_0/payloadTargetVis",
            collision_enabled=False
        )
        kit_utils.set_rigid_body_properties(
            "/World/envs/env_0/payloadTargetVis",
            disable_gravity=True
        )

        return ["/World/defaultGroundPlane"]

    def _reset_idx(self, env_ids: torch.Tensor):
        pos, rot = self.init_poses

        self.group._reset_idx(env_ids)
        self.group.set_world_poses(pos[env_ids], rot[env_ids], env_ids)
        self.group.set_velocities(self.init_velocities[env_ids], env_ids)

        self.group.set_joint_positions(self.init_joint_pos[env_ids], env_ids)
        self.group.set_joint_velocities(self.init_joint_vel[env_ids], env_ids)

        payload_target_pos = torch.rand(len(env_ids), 3, device=self.device)

        payload_target_rpy = torch.zeros(len(env_ids), 3, device=self.device)
        payload_target_rpy[..., 2] = (
            torch.pi * 2 * torch.rand(len(env_ids), device=self.device)
        )

        payload_target_rot = torch_utils.quat_from_euler_xyz(
            *payload_target_rpy.unbind(-1)
        )

        self.payload_target_pos[env_ids] = payload_target_pos
        self.payload_target_heading[env_ids] = torch_utils.quat_axis(
            payload_target_rot, axis=0
        )

        payload_target_pose = (payload_target_pos + self.envs_positions[env_ids], payload_target_rot)
        self.payload_target_visual.set_world_poses(*payload_target_pose, env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)

    def _compute_state_and_obs(self):
        drone_states = self.drone.get_state()
        drone_pos, drone_rot, drone_vels = drone_states[..., :13].split([3, 4, 6], dim=-1)
        payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_vels = self.payload.get_velocities()

        drone_relative_pos = functorch.vmap(cpos)(drone_pos, drone_pos)
        drone_relative_pos = functorch.vmap(off_diag)(drone_relative_pos)
        drone_pdist = torch.norm(drone_relative_pos, dim=-1, keepdim=True)

        payload_relative_pos_drone = payload_pos.unsqueeze(1) - drone_pos
        payload_relative_pos_target = self.payload_target_pos - payload_pos

        payload_heading: torch.Tensor = torch_utils.quat_axis(payload_rot, axis=0)
        
        payload_state = torch.cat(
            [
                payload_relative_pos_target,  # 3
                payload_rot,  # 4
                payload_vels,  # 6
                payload_heading,  # 3
            ],
            dim=-1,
        ).unsqueeze(1)

        obs = TensorDict(
            {
                "self": drone_states.unsqueeze(2),  # [num_envs, drone.n, 1, *]
                "others": torch.cat(
                    [
                        drone_relative_pos,
                        drone_pdist,
                    ],
                    dim=-1,
                ),  # [num_envs, drone.n, drone.n-1, *]
                "payload": torch.cat(
                    [
                        payload_relative_pos_drone,  # 3
                        payload_state.expand(-1, self.drone.n, -1)
                    ],
                    dim=-1,
                ).unsqueeze(2),  # [num_envs, drone.n, 1, 19]
            },
            [self.num_envs, self.drone.n],
        )

        state = TensorDict(
            {
                "drones": torch.cat(
                    [-payload_relative_pos_drone, drone_states[..., 3:]], dim=-1
                ),  # [num_envs, drone.n, *]
                "payload": payload_state,  # [num_envs, 1, 16]
            },
            self.num_envs,
        )

        return TensorDict({"drone.obs": obs, "drone.state": state}, self.num_envs)

    def _compute_reward_and_done(self):
        payload_pos, payload_rot = self.get_env_poses(self.payload.get_world_poses())
        payload_heading = torch_utils.quat_axis(payload_rot, axis=0)
        drone_pos, drone_rot = self.get_env_poses(self.drone.get_world_poses())

        distance = torch.norm(
            self.payload_target_pos - payload_pos, dim=-1, keepdim=True
        )
        reward = torch.zeros(self.num_envs, self.drone.n, 2, device=self.device)
        reward[..., 0] = -torch.square(distance)
        reward[..., 1] = (payload_heading * self.payload_target_heading).sum(-1, keepdim=True)

        self._tensordict["return"] += reward
        done = (self.progress_buf >= self.max_eposode_length).unsqueeze(-1) | (
            drone_pos[..., 2] < 0.2
        ).all(-1, keepdim=True)
        return TensorDict(
            {
                "reward": {"drone.reward": reward},
                "return": self._tensordict["return"],
                "done": done,
            },
            self.batch_size,
        )
