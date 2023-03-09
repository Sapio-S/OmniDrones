import torch
import numpy as np
import functorch
from torchrl.data import UnboundedContinuousTensorSpec, CompositeSpec
from tensordict.tensordict import TensorDict, TensorDictBase

import omni.isaac.core.utils.torch as torch_utils
from omni.isaac.core.objects import VisualSphere, DynamicSphere, FixedCuboid, FixedCylinder, VisualCuboid, DynamicCuboid, DynamicCone
from omni.isaac.core.prims import RigidPrimView, GeometryPrimView
from omni_drones.envs.isaac_env import IsaacEnv, AgentSpec
from omni_drones.robots.config import RobotCfg
from omni_drones.robots.drone import MultirotorBase
import omni_drones.utils.kit as kit_utils

# drones on land by default
# only cubes are available as walls
# clip state as walls

class Prey(IsaacEnv):
    
    def __init__(self, cfg, headless):
        super().__init__(cfg, headless)
        self.drone.initialize()
        # self.wall = GeometryPrimView(prim_paths_expr="/World/envs/.*/wall_*")
        self.target = RigidPrimView(prim_paths_expr="/World/envs/.*/target")
        # self.wall.initialize()
        self.target.initialize()
        self.target.post_reset()
        self.target_init_vel = self.target.get_velocities(clone=True)
        self.env_ids = torch.from_numpy(np.arange(0,cfg.env.num_envs))
        self.env_width = cfg.env.env_spacing/2.0
        self.progress_buf2 = self.progress_buf*1.0
        self.caught = self.progress_buf2.unsqueeze(-1)*0
        self.init_poses = self.drone.get_env_poses(clone=True)
        
        self.agent_spec["drone"] = AgentSpec(
            "drone", self.num_agents, 
            UnboundedContinuousTensorSpec(26, device=self.device),
            UnboundedContinuousTensorSpec(7, device=self.device),
            UnboundedContinuousTensorSpec(1).to(self.device),
        )
        
        self.vels = self.drone.get_velocities()
        self.init_pos_scale = torch.tensor([2., 2., 0], device=self.device) 
        self.init_pos_offset = torch.tensor([-1., -1., 0], device=self.device)

    def _design_scene(self):
        self.num_agents = 3
        cfg = RobotCfg()
        cfg.rigid_props.max_linear_velocity = 0.5
        self.drone: MultirotorBase = MultirotorBase.REGISTRY["Crazyflie"](cfg=cfg)
        # self.drone: MultirotorBase = MultirotorBase.REGISTRY["Firefly"](cfg=cfg)

        self.target_pos = torch.tensor([[0., 0.05, 0.5]], device=self.device)
        self.obstacle_pos = torch.tensor([[1.0, 0.5, 0.5]], device=self.device)

        #other prims can't even move in FixedCuboid
        DynamicSphere(
            prim_path="/World/envs/env_0/target",
            name="target",
            translation=self.target_pos,
            radius=0.05,
            # height=0.1,
            color=torch.tensor([1., 0., 0.]),
        )
        
        #without inner collision
        # FixedCuboid(
        #     prim_path="/World/envs/env_0/wall",
        #     name="obstacle",
        #     translation=torch.tensor([[0, 0, 1]]),
        #     # orientation=torch.tensor([1., 0., 0., 0.]),
        #     color=torch.tensor([0., 0.1, 0.]),
        #     size = 2,
        #     visible = 1, #air wall
        # )

        
        kit_utils.set_nested_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            max_linear_velocity=1.5,
        )

        kit_utils.set_rigid_body_properties(
            prim_path="/World/envs/env_0/target",
            disable_gravity=True
        )        

        kit_utils.create_ground_plane(
            "/World/defaultGroundPlane",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        )
        n = self.num_agents
        for i in range(n):
            translation = torch.zeros(n, 3)
            translation[:, 0] = i 
            translation[:, 1] = torch.arange(n)
            translation[:, 2] = 0.5
        self.drone.spawn(n, translation) #to make n drones
        return ["/World/defaultGroundPlane"]
    
    def _reset_idx(self, env_ids: torch.Tensor):
        n = self.num_agents
        _, rot = self.init_poses
        self.drone._reset_idx(env_ids)
        pos = torch.rand(len(env_ids), n, 3, device=self.device) * self.init_pos_scale + self.init_pos_offset
        self.drone.set_env_poses(pos, rot[env_ids], env_ids)
        self.drone.set_velocities(torch.zeros_like(self.vels[env_ids]), env_ids)
        
        self.target.set_world_poses((self.envs_positions + self.target_pos)[env_ids], indices=env_ids)
        self.target.set_velocities(self.target_init_vel[env_ids], indices=env_ids)

    def _pre_sim_step(self, tensordict: TensorDictBase):
        actions = tensordict["drone.action"]
        self.effort = self.drone.apply_action(actions)
        self.target.apply_forces(self._get_dummy_policy_prey())

        # restriction on (x,y)
        agent_pos, agent_rot = self.drone.get_env_poses()
        target_pos, target_rot = self.target.get_world_poses()
        target_pos -= self.envs_positions
        agent_pos2 = agent_pos*1.0
        target_pos2 = target_pos*1.0
        agent_pos2[...,:2] = agent_pos[...,:2].clamp(-self.env_width, self.env_width)
        agent_reverse = 1-(agent_pos != agent_pos2)*2.0
        target_pos2[...,:2] = target_pos[...,:2].clamp(-self.env_width, self.env_width)
        target_reverse = 1-(target_pos != target_pos2)*2.0
        target_pos2[...,2] = 0.5
        
        self.drone.set_env_poses(agent_pos2, agent_rot, self.env_ids)
        self.target.set_world_poses(target_pos2 + self.envs_positions, target_rot, self.env_ids)

        # bounce
        agent_vel = self.drone.get_velocities()
        agent_vel[..., :3] *= agent_reverse
        target_vel = self.target.get_velocities()
        target_vel[..., :3] *= target_reverse
        target_vel[..., 2] = 0 # z and dz/dt must be both restricted
        self.drone.set_velocities(agent_vel, self.env_ids)
        self.target.set_velocities(target_vel, self.env_ids)

    
    def _compute_state_and_obs(self):
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        drone_state = self.drone.get_state()
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        obs = torch.cat((drone_state, prey_pos),-1)
        
        return TensorDict({
            "drone.obs": obs
        }, self.batch_size)
    
    def _compute_reward_and_done(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        target_dist = torch.norm(pos-prey_pos, dim=-1)
        # pos_reward = 1.0 / (1.0 + torch.square(target_dist))
        pos_reward = - target_dist
        # pos_reward = torch.exp(-target_dist)
        # pos_reward = 1.0/target_dist # prone to over-tilt
        catch_reward = (target_dist < 0.1) * 1.0


        coll = (pos[..., :2] > self.env_width-0.1) | (pos[..., :2] < -self.env_width+0.1)
        coll_reward = (coll[..., 0] | coll[..., 1] ) * (-1.0)

        # reward = pos_reward + catch_reward + coll_reward
        reward = catch_reward
# 
        # reward = catch_reward - target_dist
        self._tensordict["drone.return"] += reward.unsqueeze(-1)

        self.progress_buf2 = self.progress_buf2 * (self.progress_buf > 0) + 1
        done  = (
            (self.progress_buf2 >= self.max_eposode_length).unsqueeze(-1)*1.0
        )
        
        caught = (catch_reward > 0) * 1.0
        self.caught = (self.progress_buf.unsqueeze(-1) > 0) * (self.caught + caught)

        #done should be integrated
        return TensorDict({
            "reward": {
                "drone.reward": reward.unsqueeze(-1)
            },
            "done": done.any(-1),
            "caught": caught.any(-1)
        }, self.batch_size)

    
    def _get_dummy_policy_prey(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        dist_pos = torch.norm(prey_pos - pos,dim=-1).unsqueeze(1).expand(-1,-1,3)
        orient = (prey_pos - pos)/dist_pos
        b = 1
        force = 0.1*(orient*1.0/dist_pos).sum(-2)
        force += b*torch.tensor([1.,0.,0.], device=self.device)/(1e-8+prey_state[...,0]+self.env_width).unsqueeze(-1)
        force += b*torch.tensor([-1.,0.,0.], device=self.device)/(1e-8-prey_state[...,0]+self.env_width).unsqueeze(-1)
        force += b*torch.tensor([0.,1.,0.], device=self.device)/(1e-8+prey_state[...,1]+self.env_width).unsqueeze(-1)
        force += b*torch.tensor([0.,-1.,0.], device=self.device)/(1e-8-prey_state[...,1]+self.env_width).unsqueeze(-1)
        return force
    
    def _get_dummy_policy_drone(self):
        pos, rot = self.drone.get_env_poses(False)
        prey_state = self.target.get_world_poses()[0] - self.drone._envs_positions.squeeze(1)
        prey_pos = prey_state.unsqueeze(1).expand(-1,self.num_agents,-1)
        dist_pos = torch.norm(prey_pos - pos,dim=-1).unsqueeze(1).expand(-1,-1,3)
        _vel = (prey_pos - pos)/dist_pos
        _pos = _vel + pos
        return _pos, _vel