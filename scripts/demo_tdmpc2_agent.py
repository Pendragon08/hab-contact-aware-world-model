"""Versioned finite-horizon, demonstration-assisted TD-MPC2 extension.

Pinned vendor files remain unchanged. Offline Q targets are recorded behavior
returns, not claims of optimal values. Shadow actions never enter dynamics.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(os.environ['HAB_ROOT'])/'vendor_tdmpc2'))
from tdmpc2 import TDMPC2
from common import math

MOTION = [i for i in range(11) if i != 7]


class DemoReplay:
    def __init__(self, payload, batch, horizon):
        self.batch, self.horizon = batch, horizon
        self.mean = torch.tensor(payload['mean'], device='cuda')
        self.std = torch.tensor(payload['std'], device='cuda')
        self.base = {k: torch.tensor(payload[k], device='cuda') for k in
                     ['observations', 'actions', 'rewards', 'terminal', 'monte_carlo_returns']}
        self.online = []
        s = payload['successful_teacher_episode']
        c = payload['correction_episode']
        self.actor_groups = []
        for mask in (s, c):
            self.actor_groups.append((torch.tensor(payload['observations'][mask, :-1].reshape(-1,45), device='cuda'),
                                      torch.tensor(payload['actor_labels'][mask].reshape(-1,11), device='cuda')))

    def normalize(self, obs):
        return (obs-self.mean)/self.std

    def actor_sample(self):
        states, actions = [], []
        # Exactly half successful demonstrations, half training-only corrections.
        for obs, act in self.actor_groups:
            idx = torch.randint(len(act), (self.batch//2,), device='cuda')
            states.append(obs[idx]); actions.append(act[idx])
        return self.normalize(torch.cat(states)), torch.cat(actions)

    def add(self, observations, actions, rewards):
        assert np.shape(observations) == (201,45) and np.shape(actions) == (200,11)
        assert np.shape(rewards) == (200,)
        self.online.append(tuple(torch.tensor(v, dtype=torch.float32, device='cuda')
                                 for v in (observations, actions, rewards)))

    def sample(self, pretrain=False):
        n = self.batch if pretrain or not self.online else self.batch//2
        h = self.horizon
        idx = torch.randint(100,(n,),device='cuda')
        starts = torch.randint(201-h,(n,),device='cuda')
        offsets = starts[None,:] + torch.arange(h+1,device='cuda')[:,None]
        o = self.base['observations'][idx[None,:],offsets]
        a = self.base['actions'][idx[None,:],offsets[:-1]]
        r = self.base['rewards'][idx[None,:],offsets[:-1],None]
        d = self.base['terminal'][idx[None,:],offsets[:-1],None]
        ret = self.base['monte_carlo_returns'][idx[None,:],offsets[:-1],None]
        if n < self.batch:
            e = torch.randint(len(self.online),(self.batch-n,)).tolist()
            st = torch.randint(201-h,(self.batch-n,)).tolist()
            more = [self.online[i] for i in e]
            o = torch.cat([o,torch.stack([x[0][s:s+h+1] for x,s in zip(more,st)],1)],1)
            a = torch.cat([a,torch.stack([x[1][s:s+h] for x,s in zip(more,st)],1)],1)
            r = torch.cat([r,torch.stack([x[2][s:s+h] for x,s in zip(more,st)],1).unsqueeze(-1)],1)
            dm = torch.tensor(st,device='cuda')[None,:]+torch.arange(h,device='cuda')[:,None] == 199
            d = torch.cat([d,dm.unsqueeze(-1)],1)
            ret = None
        return self.normalize(o), a, r, d, ret


class DemoTDMPC2(TDMPC2):
    def __init__(self, cfg, replay, settings):
        super().__init__(cfg)
        self.replay = replay
        self.settings = settings
        self.remaining = 200
        assert abs(self.discount-settings['discount']) < 1e-12

    @torch.no_grad()
    def choose(self, raw, step, mpc, evaluation):
        self.remaining = 200-step
        assert 1 <= self.remaining <= 200
        self.cfg.mpc = mpc
        return self.act(self.replay.normalize(raw), t0=step==0, eval_mode=evaluation)

    @torch.no_grad()
    def _estimate_value(self, z, actions, task):
        # Planner rewards and terminal value share the training time-limit semantics.
        k = min(self.cfg.horizon, self.remaining)
        total = torch.zeros(*z.shape[:-1],1,device=z.device)
        discount = 1.
        for t in range(k):
            total += discount*math.two_hot_inv(self.model.reward(z,actions[:,t],task),self.cfg)
            z = self.model.next(z,actions[:,t],task)
            discount *= self.discount
        if self.remaining > k:
            total += discount*self.model.Q(z,self.model.pi(z,task)[1],task,return_type='avg')
        assert torch.isfinite(total).all(), 'Non-finite MPC scores'
        return total

    def supervised_actor(self, detached=False):
        obs, action = self.replay.actor_sample()
        z = self.model.encode(obs,None)
        if detached:
            z = z.detach()
        mu, _, _, log_std = self.model.pi(z,None)
        motion = F.mse_loss(mu[:,MOTION],action[:,MOTION])
        gripper = F.mse_loss(mu[:,7],action[:,7])
        std_loss = (log_std-np.log(self.settings['prior_std'])).square().mean()
        return motion+gripper, std_loss

    def finite_target(self, reward, terminal, next_q):
        return reward + self.discount*(~terminal).float()*next_q

    def update_demo(self, pretrain):
        obs, action, reward, terminal, returns = self.replay.sample(pretrain)
        self.model.eval()
        with torch.no_grad():
            targets_z = self.model.encode(obs[1:],None)
            if pretrain:
                target_q = returns
            else:
                next_action = self.model.pi(targets_z,None)[1]
                q = self.model.Q(targets_z,next_action,None,return_type='min',target=True)
                target_q = self.finite_target(reward,terminal,q)
        self.model.train()
        self.optim.zero_grad(set_to_none=True)
        self.pi_optim.zero_grad(set_to_none=True)
        z = self.model.encode(obs[0],None)
        zs = [z]
        consistency, reward_loss, value_loss = 0.,0.,0.
        for t in range(self.cfg.horizon):
            pred_r = self.model.reward(z,action[t],None)
            qs = self.model.Q(z,action[t],None,return_type='all')
            reward_loss += self.cfg.rho**t*math.soft_ce(pred_r,reward[t],self.cfg).mean()
            value_loss += self.cfg.rho**t*sum(math.soft_ce(q,target_q[t],self.cfg).mean() for q in qs)/self.cfg.num_q
            z = self.model.next(z,action[t],None)
            consistency += self.cfg.rho**t*F.mse_loss(z,targets_z[t])
            zs.append(z)
        h = self.cfg.horizon
        consistency, reward_loss, value_loss = consistency/h,reward_loss/h,value_loss/h
        loss = self.cfg.consistency_coef*consistency + self.cfg.reward_coef*reward_loss + self.cfg.value_coef*value_loss
        bc, std_loss = self.supervised_actor(detached=False) if pretrain else (None,None)
        if pretrain:
            loss = loss + self.settings['pretrain_bc_coef']*bc + self.settings['std_coef']*std_loss
        assert torch.isfinite(loss), 'Non-finite training loss'
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.cfg.grad_clip_norm,error_if_nonfinite=True)
        self.optim.step()
        if pretrain:
            self.pi_optim.step()
            pi_loss = bc
        else:
            self.pi_optim.zero_grad(set_to_none=True)
            self.model.track_q_grad(False)
            try:
                detached_zs = torch.stack(zs).detach()
                _,pi,log_pi,_ = self.model.pi(detached_zs,None)
                qs = self.model.Q(detached_zs,pi,None,return_type='avg')
                self.scale.update(qs[0])
                weights = self.cfg.rho**torch.arange(len(qs),device='cuda')
                rl_loss = ((self.cfg.entropy_coef*log_pi-self.scale(qs)).mean((1,2))*weights).mean()
                bc,std_loss = self.supervised_actor(detached=True)
                pi_loss = rl_loss + self.settings['online_bc_coef']*bc + self.settings['std_coef']*std_loss
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(),self.cfg.grad_clip_norm,error_if_nonfinite=True)
                self.pi_optim.step()
            finally:
                self.model.track_q_grad(True)
        self.model.soft_update_target_Q()
        self.model.eval()
        values = {'loss':loss,'consistency':consistency,'reward_loss':reward_loss,'value_loss':value_loss,
                  'actor_loss':pi_loss,'bc_loss':bc,'std_loss':std_loss,'grad_norm':grad}
        out = {k:float(v.detach()) for k,v in values.items()}
        assert all(np.isfinite(v) for v in out.values()),out
        return out

    def checkpoint(self, step, updates):
        return {'model':self.model.state_dict(),'optim':self.optim.state_dict(),
                'pi_optim':self.pi_optim.state_dict(),'scale':self.scale.state_dict(),
                'mean':self.replay.mean,'std':self.replay.std,'online_step':step,'updates':updates,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'numpy_rng':np.random.get_state()}
