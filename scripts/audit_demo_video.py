"""Validate the complete concatenated demo and save a contact sheet for review."""
import json
import os
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image

ROOT = Path(os.environ.get("HAB_ROOT", Path(__file__).resolve().parents[1]))
report = ROOT / "reports/contact_v1"
examples = json.loads((report / "example_selection.json").read_text(encoding="utf-8"))["examples"]
selected = list(dict.fromkeys(i for i in examples.values() if i is not None))
expected = len(selected) * 201
reader = imageio.get_reader(ROOT / "artifacts/contact_v1/comparison_demo.mp4")
fps = reader.get_meta_data()["fps"]
assert fps == 20
samples = []
count = 0
for index, frame in enumerate(reader):
    assert frame.shape == (592, 1536, 3)
    if index % 201 == 100:
        samples.append(Image.fromarray(frame))
    count += 1
reader.close()
assert count == expected and len(samples) == len(selected)
sheet = Image.new("RGB", (1536, 592 * len(samples)), "white")
for index, sample in enumerate(samples):
    sheet.paste(sample, (0, 592 * index))
sheet.save(report / "demo_preview.png")
result = {
    "passed": True,
    "frames_decoded": count,
    "fps": fps,
    "duration_seconds": count / fps,
    "frame_shape": [592, 1536, 3],
    "selected_pair_ids_in_order": selected,
    "preview_frame_within_each_clip": 100,
}
(report / "demo_video_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result))
