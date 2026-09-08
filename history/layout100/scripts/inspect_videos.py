import json,os
from pathlib import Path
import imageio.v2 as imageio
root=Path(os.environ['HAB_ROOT']); folder=root/'results/pick_val_seed0_4_v2'
checks=[]
for p in sorted(folder.glob('*.mp4')):
    reader=imageio.get_reader(p)
    meta=reader.get_meta_data(); n=reader.count_frames()
    assert n==201,(p,n)
    checks.append({'file':p.name,'frames':n,'fps':meta['fps'],'duration':meta['duration'],'size':list(meta['size']),'bytes':p.stat().st_size})
    if '000' in p.stem:
        for index in [0,29,200]: imageio.imwrite(folder/f'preview_seed0_frame{index:03d}.png',reader.get_data(index))
    reader.close()
(folder/'video_validation.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
