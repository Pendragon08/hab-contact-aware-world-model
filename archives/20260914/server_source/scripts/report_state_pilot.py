"""Independent consistency audit for the bounded state-learning runs."""
import csv,json,os
from pathlib import Path
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(os.environ['HAB_ROOT']);DEST=ROOT/'reports/state_pilot'
def save(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
def check_tensors(x):
    if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():check_tensors(v)
    elif isinstance(x,(list,tuple)):
        for v in x:check_tensors(v)

def main():
    DEST.mkdir(parents=True,exist_ok=True);summaries={};logs={};beginnings={};audits={}
    for algo in ['tdmpc2','sac']:
        root=ROOT/'artifacts'/f'state_{algo}_pilot_v1'
        summary=json.loads((root/'summary.json').read_text());provenance=json.loads((root/'provenance.json').read_text())
        assert summary['completed'] and summary['parameters_changed'] and summary['finite_updates']
        assert summary['train_steps']==10000 and summary['updates']==9000 and summary['eval_steps']==4000
        assert provenance['initial_parameter_sha256']!=summary['final_parameter_sha256']
        assert not set(provenance['train_layouts']) & {69,74,79,84,89}
        updates=[json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()]
        assert len(updates)==9000 and all(r['update']==i+1 and r['step']==i+1001 for i,r in enumerate(updates))
        assert all(np.isfinite(float(v)) for r in updates for v in r.values())
        for ep in range(50):
            data=np.load(root/f'train_{ep:03d}.npz');meta=json.loads((root/f'train_{ep:03d}.json').read_text())
            assert data['observations'].shape==(201,provenance['state_dim']) and data['actions'].shape==(200,11)
            assert data['rewards'].shape==(200,) and meta['split']=='train' and meta['layout'] in provenance['train_layouts']
            assert len(meta['steps'])==200 and all(np.isfinite(data[k]).all() for k in data.files)
            np.testing.assert_allclose(data['rewards'],[s['reward'] for s in meta['steps']],rtol=1e-6,atol=1e-7)
            key=('train',ep)
            if key in beginnings:np.testing.assert_array_equal(beginnings[key],data['observations'][0])
            else:beginnings[key]=data['observations'][0].copy()
        evals=[]
        for ep in range(20):
            data=np.load(root/f'eval_{ep:03d}.npz');meta=json.loads((root/f'eval_{ep:03d}.json').read_text())
            assert meta['split']=='val-development' and len(meta['steps'])==200
            assert meta['summary']['success_once']==any(e['success'] for e in meta['steps'])
            assert meta['summary']['success_at_end']==meta['steps'][-1]['success']
            assert data['observations'].shape==(201,provenance['state_dim']) and data['actions'].shape==(200,11)
            assert all(np.isfinite(data[k]).all() for k in data.files)
            key=('eval',ep)
            if key in beginnings:np.testing.assert_array_equal(beginnings[key],data['observations'][0])
            else:beginnings[key]=data['observations'][0].copy()
            reader=imageio.get_reader(root/f'eval_{ep:03d}.mp4');frames=sum(1 for _ in reader);reader.close();assert frames==201
            evals.append(meta['summary'])
        assert summary['success_once_count']==sum(e['success_once'] for e in evals)
        assert summary['success_at_end_count']==sum(e['success_at_end'] for e in evals)
        check_tensors(torch.load(root/'policy.pt',map_location='cpu',weights_only=False))
        summaries[algo]={**summary,'trainable_model_parameters':provenance['trainable_model_parameters'],'state_dim':provenance['state_dim']}
        logs[algo]=updates;audits[algo]={'passed':True,'train_episodes':50,'train_steps':10000,'update_records':9000,'eval_episodes':20,'videos_decoded':20,'finite_checkpoint':True}
    save(DEST/'summary.json',summaries);save(DEST/'audit.json',{'algorithms':audits,'matching_initial_state_vectors':70,
         'note':'State vectors match across algorithms; not arbitrary simulator/controller restoration or final research test.'})
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,algo,keys in [(axes[0],'tdmpc2',['consistency_loss','reward_loss','value_loss','total_loss']),
                         (axes[1],'sac',['q_loss'])]:
        rr=logs[algo];x=np.array([r['step'] for r in rr]).reshape(-1,100).mean(1)
        for k in keys:
            y=np.array([r[k] for r in rr]).reshape(-1,100).mean(1)
            ax.plot(x,y,label=k)
        ax.set_title(algo.upper()+' pilot (100-update means)');ax.set_xlabel('Training environment steps');ax.set_ylabel('Loss');ax.legend()
    fig.tight_layout();fig.savefig(DEST/'training.png',dpi=160);fig.savefig(DEST/'training.pdf');plt.close(fig)
    lines=['# 状态输入训练可行性验证','',
           '本轮为每算法 10,000 步的工程 pilot，尚未获得收敛或最终测试证据。训练使用 train，评估使用 val-development。',
           '', '|算法|训练步|梯度更新|验证曾成功|验证结束成功|训练时间|PyTorch 峰值分配显存|', '|---|---:|---:|---:|---:|---:|---:|']
    for a,s in summaries.items():
        lines.append(f"|{a}|{s['train_steps']}|{s['updates']}|{s['success_once_count']}/20|{s['success_at_end_count']}/20|{s['training_wall_seconds']/60:.2f} 分钟|{s['torch_peak_allocated_mib']:.1f} MiB|")
    lines+=['','![训练损失](training.png)','','两算法各 50 个训练回合和 20 个验证回合；初始状态向量在相同种子下逐一匹配。全部 18,000 条梯度更新记录有限，模型参数发生真实变化，检查点可读取。',
            '', 'PyTorch 峰值不包含 PhysX、渲染和 CUDA 上下文；整卡资源用量见资源采样日志。部分运行与接触对照共享 GPU1，时间反映实际运行条件，不能当作独占硬件吞吐。',
            '', 'TD-MPC2 采用上游 5M 档架构和规划器；本项目适配 HAB 状态接口、11 个有效动作通道和回合内回放。SAC 使用相同状态和交互预算，但网络规模不同。损失下降证明优化在进行，不能替代任务成功率。',
            '', '时间上限按 continuing-task 设定自举；状态包含当前步比例，本 pilot 尚未比较有限时域终止处理的影响。',
            '', '风险模块进入条件：先建立稳定完成任务的基线、充分训练预算及独立测试协议。本轮短训不足以声称风险规划有效，也不能据此否定世界模型。后续训练规模应根据这里测得的实际成本规划。']
    (DEST/'报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('STATE_REPORT_COMPLETE',json.dumps(summaries),flush=True)

if __name__=='__main__':main()
