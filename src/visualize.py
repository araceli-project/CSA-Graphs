import os

import numpy as np
import cv2
import torch

PERSON_COLORS = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Cyan
    (255, 0, 255), # Magenta
    (0, 255, 255), # Yellow
    (128, 0, 128), # Purple
    (128, 128, 0), # Olive
    (0, 128, 128), # Teal
    (128, 128, 128) # Gray
]

def visualize_poses(poses_file="poses.npz", image_paths=None, output_dir="output"):
    
    poses_file = np.load(poses_file, allow_pickle=True)
    poses = poses_file['poses']
    print(poses.item().keys())
    print(poses.item()["poses_xy"])
    
    if image_paths is None:
        print("Please provide image_paths to visualize")
        return

    for idx, image_path in enumerate(image_paths):
        if idx >= len(poses.item()["poses_xy"]):
            break
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
        
        pose_data = poses.item()["poses_xy"][idx]
        if pose_data.size > 0:
            for person_keypoints in pose_data:
                for keypoint in person_keypoints:
                    x, y = int(keypoint[0]), int(keypoint[1])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        
        cv2.imwrite(f"pose_{image_path}", image)

def visualize_poses_graphs(poses_file="graph_data.pt", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    graph_data = torch.load(poses_file, weights_only=False)[:1630]
    for idx, geometric_data in enumerate(graph_data):
        if geometric_data.x.shape[0] == 0:
            continue

        # get resolution
        max_x = int(np.max(geometric_data.x[:, 0]).item())
        max_y = int(np.max(geometric_data.x[:, 1]).item())
        image = np.zeros((max_y + 50, max_x + 50, 3), dtype=np.uint8)

        for key_idx, keypoint in enumerate(geometric_data.x):
            x, y = int(keypoint[0].item()), int(keypoint[1].item())
            cv2.circle(image, (x, y), 5, PERSON_COLORS[key_idx//17 % len(PERSON_COLORS)], -1)

        for edge_idx, edge in enumerate(geometric_data.edge_index.T):
            start, end = edge
            start_point = (int(geometric_data.x[start][0].item()), int(geometric_data.x[start][1].item()))
            end_point = (int(geometric_data.x[end][0].item()), int(geometric_data.x[end][1].item()))
            cv2.line(image, start_point, end_point, PERSON_COLORS[edge_idx//18 % len(PERSON_COLORS)], 2)
        cv2.imwrite(f"{output_dir}/pose_graph_{idx}.png", image)


if __name__ == "__main__":
    visualize_poses_graphs(poses_file="graph_data.pt")