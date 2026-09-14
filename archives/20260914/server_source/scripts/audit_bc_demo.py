"""Decode the entire BC demo and preserve one middle frame per selected example."""
import hashlib
import json
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image

root = Path(__file__).resolve().parents[1]
report = root/'reports/bc_v1_2'
selection = json.loads((report/'demo_selection.json').read_text())
video = report/'teacher_student_demo.mp4'
reader = imageio.get_reader(video)
fps = reader.get_meta_data()['fps']; assert fps == 20
samples = []; count = 0
for i, frame in enumerate(reader):
    assert frame.shape == (576, 1024, 3)
    if i % 201 == 100: samples.append(Image.fromarray(frame))
    count += 1
reader.close()
assert count == selection['frames'] == 201*len(selection['indices'])
assert len(samples) == len(selection['indices'])
sheet = Image.new('RGB',(1024,576*len(samples)))
for i, sample in enumerate(samples): sheet.paste(sample,(0,576*i))
sheet.save(report/'demo_preview.png')
result = {'passed':True,'frames_decoded':count,'fps':fps,'duration_seconds':count/fps,
          'shape':[576,1024,3],'selection':selection,'preview_frame_per_clip':100,
          'video_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
          'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(report/'demo_video_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('BC_DEMO_AUDIT_PASS',json.dumps(result))
