import torch
import hydra
import os
import time
from tensordict import TensorDict
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from tqdm import tqdm
from omegaconf import OmegaConf
from setproctitle import setproctitle
from omni_drones import CONFIG_PATH, init_simulation_app
from omni_drones.learning.collectors import SyncDataCollector
from omni_drones.utils.wandb import init_wandb
from omni_drones.utils.envs.transforms import LogOnEpisode

from functorch import vmap

@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg):
    cfg.use_load = 0
    cfg.wandb.mode = 'disabled'
    # cfg.env.num_envs = 16
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    print(cfg.wandb)
    run = init_wandb(cfg)
    setproctitle(run.name)
    simulation_app = init_simulation_app(cfg)    
    print(OmegaConf.to_yaml(cfg))
    from omni_drones.envs import Prey
    from omni_drones.learning.mappo import MAPPOPolicy

    env = Prey(cfg, headless=cfg.headless)
    agent_spec = env.agent_spec["drone"]

    ppo = MAPPOPolicy(cfg.algo, agent_spec, act_name="drone.control_target", device="cuda")
    if os.path.exists('model.pkl') and cfg.use_load:
        ppo.load_state_dict(torch.load('model.pkl'))
        print("Successfully load model!")

    # ppo.load_state_dict()
    controller = env.drone.default_controller(
        env.drone.dt, 9.81, env.drone.params
    ).to(env.device)

    def policy(tensordict: TensorDict):
        state = tensordict["drone.obs"]
        tensordict = ppo(tensordict)
        relative_state = tensordict["drone.obs"][..., :13]
        control_target = tensordict["drone.control_target"]
        controller_state = tensordict.get("controller_state", TensorDict({}, state.shape[:2]))
        # _pos, _vel = env._get_dummy_policy_drone()
        # control_target[... ,:3] = _pos
        # control_target[... ,3:6] = _vel
        # control_target[... ,6] = 0    
        cmds, controller_state = vmap(vmap(controller))(relative_state, control_target, controller_state)     # len(control target)=7
        torch.nan_to_num_(cmds, 0.)
        assert not torch.isnan(cmds).any()
        tensordict["drone.action"] = cmds #command for motor
        tensordict["controller_state"] = controller_state 
        return tensordict

    logger = LogOnEpisode(
        cfg.env.num_envs,
        in_keys=["return", "progress", "success"],
        log_keys=["train/return", "train/ep_length", "train/success_rate"],
        logger_func=run.log
    )
        
    collector = SyncDataCollector(
        env, 
        policy, 
        callback=logger,
        split_trajs=False,
        frames_per_batch=env.num_envs * 8,
        device="cuda", 
        return_same_td=True,
    )
    
    pbar = tqdm(collector)
    for i, data in enumerate(pbar):
        if cfg.train:
            info = ppo.train_op(data.clone())
            run.log(info)
        pbar.set_postfix({
            "rollout_fps": collector._fps,
            "frames": collector._frames
        })
        if i%1e3==0 and i>0:
            ppo.save()
        
    simulation_app.close()

if __name__ == "__main__":
    main()