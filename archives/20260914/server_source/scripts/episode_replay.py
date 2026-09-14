"""Small pilot replay with windows that cannot cross episode boundaries."""
import torch

class EpisodeReplay:
    def __init__(self,batch_size,horizon,device='cuda'):
        self.batch_size=batch_size;self.horizon=horizon;self.device=device
        self.episodes=[]
    def add(self,obs,action,reward):
        assert len(obs)==len(action)+1==len(reward)+1
        assert len(action)>=self.horizon
        self.episodes.append(tuple(torch.as_tensor(x,device=self.device,dtype=torch.float32) for x in (obs,action,reward)))
    def sample(self):
        assert self.episodes
        # Equal 200-step episodes imply uniform windows when sampling episodes uniformly.
        eidx=torch.randint(len(self.episodes),(self.batch_size,)).tolist()
        starts=torch.randint(len(self.episodes[0][1])-self.horizon+1,(self.batch_size,)).tolist()
        o=[];a=[];r=[]
        for e,s in zip(eidx,starts):
            obs,action,reward=self.episodes[e]
            o.append(obs[s:s+self.horizon+1]);a.append(action[s:s+self.horizon]);r.append(reward[s:s+self.horizon])
        return torch.stack(o,1),torch.stack(a,1),torch.stack(r,1).unsqueeze(-1),None
    def sample_transitions(self):
        obs,a,r,_=self.sample()
        return obs[0],a[0],r[0],obs[1]
