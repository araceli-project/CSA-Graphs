# Script to transform pose data from .npz to JSON format

import json
import numpy as np

poses_file  = np.load("poses.npz", allow_pickle=True)
poses       = poses_file['poses'].item()

print("Keys in the poses dictionary:", poses.keys())
poses_xy    = poses["poses_xy"][:1630]
poses_data  = poses["poses_data"][:1630]
poses_conf  = poses["poses_conf"][:1630]
poses_xyn   = poses["poses_xyn"][:1630]
poses_boxes = poses["boxes"][:1630]

records = []
for img_idx, (xy, data, conf, xyn, boxes) in enumerate(
    zip(poses_xy, poses_data, poses_conf, poses_xyn, poses_boxes), start=1
):
    records.append({
        "image_id": img_idx,
        "xy"      : xy.tolist(),
        "data"    : data.tolist(),
        "conf"    : conf.tolist(),
        "xyn"     : xyn.tolist(),
        "boxes"   : boxes.tolist(),
    })

with open("poses.json", "w") as f:
    json.dump(records, f)

print(f"Saved {len(records)} images → poses.json")