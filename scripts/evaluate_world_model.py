"""Post-hoc 1/2/3-step reward prediction on saved development trajectories."""
import json,os,sys
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(os.environ['HAB_ROOT']);sys.path.insert(0,str(ROOT/'vendor_tdmpc2'))
from tdmpc2 import TDMPC2
from common.math import two_hot_inv

def main():
    folder=ROOT/'artifacts/state_tdmpc2_pilot_v1';out=ROOT/'reports/world_model_prediction';out.mkdir(parents=True,exist_ok=False)
    cfg=OmegaConf.load(folder/'resolved_config.yaml');agent=TDMPC2(cfg)
    agent.load(torch.load(folder/'policy.pt',map_location='cuda',weights_only=False))
    train_rewards=np.concatenate([np.load(p)['rewards'] for p in sorted(folder.glob('train_*.npz'))])
    assert len(train_rewards)==10000
    constant=float(train_rewards.mean());metrics={};traces={}
    for behavior in ['tdmpc2','sac']:
        base=ROOT/f'artifacts/state_{behavior}_pilot_v1';obs=[];actions=[];truth=[];ids=[]
        for ep in range(20):
            with np.load(base/f'eval_{ep:03d}.npz') as data:
                records=json.loads((base/f'eval_{ep:03d}.json').read_text())
                assert records['split']=='val-development'
                rewards=np.array([r['reward'] for r in records['steps']],dtype=np.float32)
                for start in range(198):
                    obs.append(data['observations'][start]);actions.append(data['actions'][start:start+3]);truth.append(rewards[start:start+3]);ids.append([ep,start])
        obs=np.stack(obs);actions=np.stack(actions);truth=np.stack(truth);prediction=[]
        with torch.no_grad():
            for begin in range(0,len(obs),256):
                z=agent.model.encode(torch.tensor(obs[begin:begin+256],device='cuda'),None)
                a=torch.tensor(actions[begin:begin+256],device='cuda');batch=[]
                for h in range(3):
                    # Future state is never fed back: all horizons unroll the learned dynamics.
                    batch.append(two_hot_inv(agent.model.reward(z,a[:,h],None),cfg).squeeze(-1).cpu().numpy())
                    z=agent.model.next(z,a[:,h],None)
                prediction.append(np.stack(batch,1))
        prediction=np.concatenate(prediction);assert prediction.shape==truth.shape==(3960,3) and np.isfinite(prediction).all()
        table=[]
        for h in range(3):
            error=prediction[:,h]-truth[:,h];reference=constant-truth[:,h]
            table.append({'horizon':h+1,'windows':len(error),'model_mae':float(np.abs(error).mean()),'model_rmse':float(np.sqrt(np.square(error).mean())),
                          'constant_mae':float(np.abs(reference).mean()),'constant_rmse':float(np.sqrt(np.square(reference).mean()))})
        metrics[behavior]=table;traces[behavior]=(truth,prediction)
        np.savez_compressed(out/f'{behavior}_prediction_data.npz',episode_and_start=np.array(ids),reward_truth=truth,reward_prediction=prediction)
    report={'purpose':'reward prediction diagnostic, not physical-state prediction or final-test evidence',
            'constant_baseline':'mean of 10,000 TD-MPC2 training rewards, never fit on evaluation trajectories',
            'constant_reward':constant,'metrics_by_behavior':metrics,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'caveat':'Overlapping windows are correlated; no independent-sample significance or causal/physical prediction claim.'}
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,behavior in zip(axes,['tdmpc2','sac']):
        table=metrics[behavior]
        ax.plot([1,2,3],[r['model_mae'] for r in table],'o-',label='Learned world model')
        ax.plot([1,2,3],[r['constant_mae'] for r in table],'s--',label='Train-mean constant')
        ax.set_xticks([1,2,3]);ax.set_xlabel('Open-loop prediction horizon');ax.set_ylabel('Reward MAE');ax.set_title(behavior.upper()+' development rollouts');ax.legend()
    fig.tight_layout();fig.savefig(out/'reward_prediction.png',dpi=160);plt.close(fig)
    lines=['# 世界模型奖励预测诊断','',
           '使用 TD-MPC2 已保存的检查点，对两种状态策略的验证轨迹做 1/2/3 步开环奖励预测。仅第一个隐状态由真实观测编码，后续隐状态通过模型和已记录动作推进，不喂入未来真实状态。',
           '', '常数参考只使用 TD-MPC2 训练奖励的平均值，不使用验证数据拟合。每种行为策略 20 回合，各有 3,960 个长度为 3 的时间窗；相邻窗口重叠，不视为独立样本。',
           '', '|行为数据|预测步长|模型 MAE|常数 MAE|模型 RMSE|常数 RMSE|','|---|---:|---:|---:|---:|---:|']
    for b,rows in metrics.items():
        for r in rows:lines.append(f"|{b}|{r['horizon']}|{r['model_mae']:.5f}|{r['constant_mae']:.5f}|{r['model_rmse']:.5f}|{r['constant_rmse']:.5f}|")
    lines+=['','![奖励预测误差](reward_prediction.png)','',
            '这是奖励预测的有限诊断；TD-MPC2 使用隐式世界模型，此处未验证物体位姿、接触力或图像的物理预测精度。即使模型误差低于常数参考，也不能替代抓取成功率或证明风险规划有效。']
    (out/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('WORLD_MODEL_PREDICTION_AUDIT',json.dumps(report),flush=True)

if __name__=='__main__':main()
