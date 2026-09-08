"""Decode all videos, create an unretimed example reel, extract original frames."""
import argparse, hashlib, json, subprocess
from pathlib import Path
import imageio.v2 as imageio
import imageio_ffmpeg

def main(root):
    folder=root/"results/pick_val_layout100_v1"
    summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
    rows=json.loads((folder/"episodes.json").read_text(encoding="utf-8"))
    checks=[]
    for row in rows:
        video=folder/row["video"]
        reader=imageio.get_reader(video)
        try:
            meta=reader.get_meta_data(); count=reader.count_frames()
            assert count==201 and meta["fps"]==20 and tuple(meta["size"])==(512,512), (video,meta,count)
            checks.append({"episode_id":row["episode_id"],"video":row["video"],"frames":count,
                           "fps":meta["fps"],"size":list(meta["size"]),"duration":meta["duration"],
                           "sha256":hashlib.sha256(video.read_bytes()).hexdigest()})
        finally: reader.close()
    examples=summary["examples"]
    concat=folder/"example_concat.txt"
    concat.write_text("".join(f"file '{(folder/rows[e['episode_id']]['video']).as_posix()}'\n" for e in examples),encoding="utf-8")
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    destination=folder/"failure_analysis_examples.mp4"
    assert not destination.exists()
    subprocess.run([ffmpeg,"-hide_banner","-loglevel","error","-f","concat","-safe","0",
                    "-i",str(concat),"-c","copy",str(destination)],check=True)
    reader=imageio.get_reader(destination)
    count=reader.count_frames()
    assert count==201*len(examples),count
    reader.close()
    extracts=[]
    for example in examples:
        ep_id=example["episode_id"]; row=rows[ep_id]
        events=[json.loads(x) for x in (folder/f"episode_{ep_id:03d}"/"steps.jsonl").read_text().splitlines()]
        # An unmodified initial, task-relevant middle, and terminal frame.
        mid=row["first_success_step"] or (row["grasp_loss_start_steps"][0] if row["grasp_loss_start_steps"] else row["first_grasp_step"]) or 100
        reader=imageio.get_reader(folder/row["video"])
        for frame_id in sorted(set([0,int(mid),200])):
            p=folder/f"example_{ep_id:03d}_frame_{frame_id:03d}.png"
            imageio.imwrite(p,reader.get_data(frame_id))
            extracts.append({"episode_id":ep_id,"category":example["category"],"frame":frame_id,"file":p.name})
        reader.close()
    (folder/"video_validation.json").write_text(json.dumps({"individual":checks,"montage_frames":count,
        "montage_seconds":count/20,"example_selection":examples,"extracted_frames":extracts},ensure_ascii=False,indent=2),encoding="utf-8")
    print("MEDIA_VALIDATED",len(checks),"videos",count,"montage frames")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",required=True)
    main(Path(p.parse_args().root))

