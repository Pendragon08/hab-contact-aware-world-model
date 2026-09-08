"""State-only HAB adapter; original task reward and success criteria unchanged."""
import numpy as np
import torch
import gymnasium as gym
from mani_skill import ASSET_DIR
from mshab.envs.make import plan_data_from_file

ACTIVE=[0,1,2,3,4,5,6,7,10,11,12]

class HABState:
    def __init__(self,split):
        assert split in ('train','val')
        root=ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange'
        data=plan_data_from_file(root/f'task_plans/tidy_house/pick/{split}/all.json')
        self.env=gym.make('PickSubtaskTrain-v0',num_envs=1,obs_mode='state',render_mode='rgb_array',sim_backend='gpu',
                          max_episode_steps=200,robot_uids='fetch',control_mode='pd_joint_delta_pos',reward_mode='normalized_dense',
                          task_plans=data.plans,scene_builder_cls=data.dataset,
                          spawn_data_fp=str(root/f'spawn_data/tidy_house/pick/{split}/spawn_data.pt'),
                          require_build_configs_repeated_equally_across_envs=False,add_event_tracker_info=True)
        self.split=split;self.steps=0
        self.layouts=sorted(self.env.unwrapped.build_config_idx_to_task_plans)
        mapping={k:tuple(v) for k,v in self.env.unwrapped.agent.controller.action_mapping.items()}
        assert mapping=={'arm':(0,7),'gripper':(7,8),'body':(8,11),'base':(11,13)},mapping

    def augment(self,obs,info):
        assert isinstance(obs,torch.Tensor) and obs.shape[0]==1
        force=info['robot_force'].reshape(1,1)/100.
        cumulative=info['robot_cumulative_force'].reshape(1,1)/5000.
        elapsed=torch.full((1,1),self.steps/200.,device=obs.device)
        x=torch.cat([obs.float(),force,cumulative,elapsed],dim=1)
        assert torch.isfinite(x).all()
        return x

    def reset(self,seed,layout=None):
        self.steps=0
        options={'reconfigure':True}
        if layout is not None:options['build_config_idxs']=[layout]
        obs,info=self.env.reset(seed=seed,options=options)
        self.last_info=info
        if layout is not None: assert list(self.env.unwrapped.build_config_idxs)==[layout]
        return self.augment(obs,info),info

    def step(self,action):
        a=torch.as_tensor(action,device=self.env.unwrapped.device,dtype=torch.float32).reshape(1,11)
        assert torch.isfinite(a).all() and a.abs().max()<=1.00001
        full=torch.zeros((1,13),device=a.device);full[:,ACTIVE]=a
        obs,reward,terminated,truncated,info=self.env.step(full)
        self.steps+=1;self.last_info=info
        # Upstream can flag early success; this continuous 200-step experiment ignores that termination.
        done=self.steps==200
        assert bool(truncated[0].item())==done
        return self.augment(obs,info),reward,done,info

    def close(self):self.env.close()
