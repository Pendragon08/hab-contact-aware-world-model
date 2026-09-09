"""Five real original-core updates on saved bounded HAB probe transitions."""
from runner import *
from tensordict import TensorDict
guard(True);verify_sources();set_seed(1)
files=sorted((HERE/'artifacts').glob('witness_*/trajectories/train/000000.npz'))
assert files
x=np.load(files[-1]);assert x['obs'].shape==(201,44)
c=OmegaConf.load(ROOT/'tdmpc2_original/upstream/tdmpc2/config.yaml');del c['defaults']
for k,v in MODEL_SIZE[5].items():c[k]=v
c.update({'obs':'state','obs_shape':{'state':[44]},'action_dim':11,'multitask':False,'task_dim':0,'episode_length':200,
          'steps':500000,'batch_size':256,'bin_size':(c.vmax-c.vmin)/(c.num_bins-1)})
a=TDMPC2(c);before=hashes(a);b=Buffer(c)
td=TensorDict({'obs':torch.from_numpy(x['obs']).float(),
 'action':torch.cat([torch.full((1,11),float('nan')),torch.from_numpy(x['action']).float()]),
 'reward':torch.cat([torch.tensor([float('nan')]),torch.from_numpy(x['reward']).float()])},batch_size=[201])
b.add(td);batch=list(b.sample());assert batch[0].shape==(4,256,44)
assert all(torch.isfinite(z).all() for z in batch if z is not None)
losses=[]
for _ in range(5):
    m={k:scalar(v) for k,v in a.update(b).items()};assert all(np.isfinite(v) for v in m.values());losses.append(m)
after=hashes(a);assert all(u!=v for u,v in zip(before,after))
d={'passed':True,'updates':5,'all_five_modules_changed':True,'replay_shape':list(batch[0].shape),'source_trajectory':str(files[-1]),'losses':losses}
save(HERE/'reports/kernel-probe.json',d);print('HAB_ORIGINAL_KERNEL_PASS',json.dumps(d))
