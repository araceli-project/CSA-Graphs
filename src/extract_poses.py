import os
import wget
import ultralytics
from tqdm import tqdm
import numpy as np
import pandas as pd

def main():

    # Model loading
    if not os.path.exists("yolo26x-pose.pt"):
        print("Downloading yolo26x-pose.pt...")
        wget.download("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt", "yolo26x-pose.pt")
    model = ultralytics.YOLO("yolo26x-pose.pt")
    print("Model loaded successfully!")

    # Getting paths 
    df = pd.read_csv("rcpd_annotation_fix.csv")

    prepend_image_dir = input("Enter the directory path where the images are stored (e.g., /path/to/images) (without the final /): ")

    image_path_list = df["filename"].tolist()
    for i in range(len(image_path_list)):
        image_path_list[i] = prepend_image_dir + image_path_list[i]

    # Setting up data structure to store poses
    poses = {}
    poses["poses_xy"] = []
    poses["poses_xyn"] = []
    poses["poses_data"] = []
    poses["poses_conf"] = []
    poses["boxes"] = []
    # Extract poses from all images
    print("Processing images...")
    for image_path in tqdm(image_path_list):
        results = model(image_path)
        for result in results:
            poses["poses_xy"].append(result.keypoints.xy.cpu().numpy())
            poses["poses_xyn"].append(result.keypoints.xyn.cpu().numpy())
            poses["poses_data"].append(result.keypoints.data.cpu().numpy())
            poses["poses_conf"].append(result.keypoints.conf.cpu().numpy())
            poses["boxes"].append(result.boxes.xyxy.cpu().numpy())
    print(f"Total images processed: {len(image_path_list)}")

    np.savez("poses", poses=poses)  # Save the poses to a .npz file

if __name__ == "__main__":
    main()