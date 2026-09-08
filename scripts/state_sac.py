"""Matched-state SAC reference for the bounded HAB engineering pilot."""
import copy
import torch
from torch import nn
from torch.distributions import Normal

def mlp(i,o):return nn.Sequential(nn.Linear(i,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,o))

class StateSAC:
    def __init__(self,obs_dim,action_dim,lr=3e-4,gamma=.975):
        self.actor=mlp(obs_dim,2*action_dim).cuda()
        self.q1=mlp(obs_dim+action_dim,1).cuda();self.q2=mlp(obs_dim+action_dim,1).cuda()
        self.t1=copy.deepcopy(self.q1).requires_grad_(False);self.t2=copy.deepcopy(self.q2).requires_grad_(False)
        self.pi_opt=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=lr)
        self.log_alpha=torch.tensor(0.,device='cuda',requires_grad=True)
        self.alpha_opt=torch.optim.Adam([self.log_alpha],lr=lr)
        self.action_dim=action_dim;self.gamma=gamma
    def distribution(self,obs):
        mu,log_std=self.actor(obs).chunk(2,-1)
        return mu,log_std.clamp(-5,2)
    def sample(self,obs):
        mu,log_std=self.distribution(obs);dist=Normal(mu,log_std.exp());z=dist.rsample();a=z.tanh()
        lp=(dist.log_prob(z)-torch.log(1-a.square()+1e-6)).sum(-1,keepdim=True)
        return a,lp
    @torch.no_grad()
    def act(self,obs,t0=False,eval_mode=False):
        if eval_mode:return self.distribution(obs)[0].tanh().cpu()
        return self.sample(obs)[0].cpu()
    def update(self,replay):
        obs,a,r,nxt=replay.sample_transitions();alpha=self.log_alpha.exp().detach()
        with torch.no_grad():
            na,nlp=self.sample(nxt); inp=torch.cat([nxt,na],-1)
            target=r+self.gamma*(torch.minimum(self.t1(inp),self.t2(inp))-alpha*nlp)
        inp=torch.cat([obs,a],-1)
        qloss=(self.q1(inp)-target).square().mean()+(self.q2(inp)-target).square().mean()
        self.q_opt.zero_grad(set_to_none=True);qloss.backward();self.q_opt.step()
        for q in (self.q1,self.q2):q.requires_grad_(False)
        pa,lp=self.sample(obs);inp=torch.cat([obs,pa],-1)
        ploss=(alpha*lp-torch.minimum(self.q1(inp),self.q2(inp))).mean()
        self.pi_opt.zero_grad(set_to_none=True);ploss.backward();self.pi_opt.step()
        for q in (self.q1,self.q2):q.requires_grad_(True)
        aloss=-(self.log_alpha*(lp.detach()-self.action_dim)).mean()
        self.alpha_opt.zero_grad(set_to_none=True);aloss.backward();self.alpha_opt.step()
        with torch.no_grad():
            for src,target_net in [(self.q1,self.t1),(self.q2,self.t2)]:
                for p,t in zip(src.parameters(),target_net.parameters()):t.lerp_(p,.01)
        return {'q_loss':float(qloss),'pi_loss':float(ploss),'alpha':float(self.log_alpha.exp()),'alpha_loss':float(aloss)}
    def checkpoint(self):
        return {'actor':self.actor.state_dict(),'q1':self.q1.state_dict(),'q2':self.q2.state_dict(),
                't1':self.t1.state_dict(),'t2':self.t2.state_dict(),'log_alpha':self.log_alpha.detach(),
                'pi_optim':self.pi_opt.state_dict(),'q_optim':self.q_opt.state_dict(),'alpha_optim':self.alpha_opt.state_dict()}
