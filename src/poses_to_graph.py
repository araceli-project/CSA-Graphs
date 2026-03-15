# Produce torch geometric graph data from the pose npz.
import numpy as np
import torch
from torch_geometric.data import Data

"""
The keypoints represent 17 parts of the body, which are in order:
    Nose
    Left Eye
    Right Eye
    Left Ear
    Right Ear
    Left Shoulder
    Right Shoulder
    Left Elbow
    Right Elbow
    Left Wrist
    Right Wrist
    Left Hip
    Right Hip
    Left Knee
    Right Knee
    Left Ankle
    Right Ankle

"""
edges_guide = np.array([
    (0, 1), (0, 2), # Nose to Eyes
    (1, 3), (2, 4), # Eyes to Ears
    (3, 5), (4, 6), # Ears to Shoulders 
    (5, 6), # Shoulders to each other
    (5, 7), (6, 8), # Shoulders to Elbows
    (7, 9), (8, 10), # Elbows to Wrists
    (5, 11), (6, 12), # Shoulders to Hips
    (11, 12), # Hips to each other
    (11, 13), (12, 14), # Hips to Knees
    (13, 15), (14, 16) # Knees to Ankles
])

# Using xy data from the pose npz
def pose_to_graph_xy(poses_file) -> list[Data]:
    poses_file = np.load(poses_file, allow_pickle=True)
    poses = poses_file['poses']
    xy = poses.item()["poses_xy"]

    graph_data_list = []
    for image in xy:
        # One graph per image with all skeletons in that image
        edges_guide_inst = edges_guide.copy()  # Copy the guide for each image
        edge_index = []
        keypoints = []
        for person in image:
            for keypoint in person:
                keypoints.append(keypoint)
            for edge in edges_guide_inst:
                edge_index.append(edge.copy())  # Append a copy of the edge
            edges_guide_inst += 17  # Shift the guide for the next person (17 keypoints per person)
        edge_index = np.array(edge_index).T  # Transpose to get shape [2, num_edges]
        keypoints = np.array(keypoints)
        graph_data = Data(x=keypoints, edge_index=edge_index)
        graph_data_list.append(graph_data)

    return graph_data_list

# Using xy and visibility data from the pose npz
def pose_to_graph_data(poses_file) -> list[Data]:
    poses_file = np.load(poses_file, allow_pickle=True)
    poses = poses_file['poses']
    xyv = poses.item()["poses_data"]
    confidence = poses.item()["poses_conf"]

    graph_data_list = []
    for image,conf_image  in zip(xyv, confidence):
        # One graph per image with all skeletons in that image
        edges_guide_inst = edges_guide.copy()  # Copy the guide for each image
        edge_index = []
        keypoints = []
        for person, conf_person in zip(image, conf_image):
            for keypoint, conf_keypoint in zip(person, conf_person):
                keypoints.append(np.concatenate((keypoint, [conf_keypoint])))  # Append confidence to the keypoint data
            for edge in edges_guide_inst:
                edge_index.append(edge.copy())  # Append a copy of the edge
            edges_guide_inst += 17  # Shift the guide for the next person (17 keypoints per person)
        edge_index = np.array(edge_index).T  # Transpose to get shape [2, num_edges]
        keypoints = np.array(keypoints)
        graph_data = Data(x=keypoints, edge_index=edge_index)
        graph_data_list.append(graph_data)

    return graph_data_list

if __name__ == "__main__":
    graph_data_list = pose_to_graph_xy("poses.npz")
    print(f"Generated {len(graph_data_list)} graph data objects.")
    print(graph_data_list[0])  # Print the first graph data for verification
    print(graph_data_list[25])  # Print the first graph data for verification
    print("Edges for the second person in the 25th image:")
    print(graph_data_list[25].edge_index)
    print("Keypoints for the second person in the 25th image:")
    print(graph_data_list[25].x)
    torch.save(graph_data_list, "graph_data.pt")