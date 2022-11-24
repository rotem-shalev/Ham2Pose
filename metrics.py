import os
import json
import random
import numpy as np
import importlib
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

import sys
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from pose_format.pose_header import PoseHeader
from pose_format.utils.reader import BufferReader
from pose_utils import pose_normalization_info, pose_hide_legs, pose_hide_low_conf
from data.tfds_dataset import flip_pose
from data.hamnosys.hamnosys import get_pose
from predict import predict_pose

PJM_FRAME_WIDTH = 1280
with open("data/pjm_left_videos.json", 'r') as f:
    PJM_LEFT_VIDEOS_LST = json.load(f)

dataset_module = importlib.import_module(f"data.hamnosys.hamnosys")

with open(dataset_module._POSE_HEADERS["openpose"], "rb") as buffer:
    pose_header = PoseHeader.read(BufferReader(buffer.read()))


# utils
def get_pose(keypoints_path: str, datum_id: str, fps: int = 25):
    pose = get_pose(keypoints_path, fps)

    if datum_id in PJM_LEFT_VIDEOS_LST:
        pose = flip_pose(pose)

    normalization_info = pose_normalization_info(pose_header)
    pose = pose.normalize(normalization_info)
    pose.focus()

    pose_hide_legs(pose)
    pose_hide_low_conf(pose)

    # Prune all leading frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Prune all trailing frames containing only zeros, almost no face, or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i][:, 25:-42].sum() > 35 and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > 0:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return pose


def masked_euclidean(point1, point2):
    if np.ma.is_masked(point2):  # reference label keypoint is missing
        return 0
    elif np.ma.is_masked(point1):  # reference label keypoint is not missing, other label keypoint is missing
        return euclidean((0, 0), point2)/2
    d = euclidean(point1, point2)
    return d


def masked_mse(trajectory1, trajectory2, confidence):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
        confidence = np.concatenate((confidence, np.zeros((diff))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return (sq_error * confidence).mean()


def mse(trajectory1, trajectory2):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return sq_error.mean()


def APE(trajectory1, trajectory2):
    if len(trajectory1) < len(trajectory2):
        diff = len(trajectory2) - len(trajectory1)
        trajectory1 = np.concatenate((trajectory1, np.zeros((diff, 2))))
    elif len(trajectory2) < len(trajectory1):
        trajectory2 = np.concatenate((trajectory2, np.zeros((len(trajectory1) - len(trajectory2), 2))))
    pose1_mask = np.ma.getmask(trajectory1)
    pose2_mask = np.ma.getmask(trajectory2)
    trajectory1[pose1_mask] = 0
    trajectory1[pose2_mask] = 0
    trajectory2[pose1_mask] = 0
    trajectory2[pose2_mask] = 0
    sq_error = np.power(trajectory1 - trajectory2, 2).sum(-1)
    return np.sqrt(sq_error).mean()


def compare_pose_videos(pose1_id, pose2_id, keypoints_path, distance_function=fastdtw):
    pose1 = get_pose(os.path.join(keypoints_path, pose1_id), pose1_id)
    pose2 = get_pose(os.path.join(keypoints_path, pose2_id), pose2_id)
    return compare_poses(pose1, pose2, distance_function=distance_function)


def get_idx2weight(max_idx):
    idx2weight = {i: 1 for i in range(9)}
    idx2weight.update({i: 1 for i in range(95, max_idx)})
    return idx2weight


def compare_poses(pose1, pose2, distance_function=fastdtw):
    # don't use legs, face for trajectory distance computations- only upper body and hands
    poses_data = get_pose_data([pose1, pose2])
    total_distance = 0
    idx2weight = get_idx2weight(pose1.body.data.shape[2])

    for keypoint_idx, weight in idx2weight.items():
        pose1_keypoint_trajectory = poses_data[0][:, :, keypoint_idx].squeeze(1)
        pose2_keypoint_trajectory = poses_data[1][:, :, keypoint_idx].squeeze(1)

        if distance_function in [mse, APE]:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory)
        elif distance_function == masked_mse:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory, pose1.body.confidence[:,
                                                                                           :, keypoint_idx].squeeze(1))
        else:
            dist = distance_function(pose1_keypoint_trajectory, pose2_keypoint_trajectory, dist=masked_euclidean)[0]
        total_distance += dist*weight
    return total_distance/len(idx2weight)


def get_pose_data(poses):
    # return relevant pose data for trajectory distance computations- only upper body and hands
    poses_data = []
    for pose in poses:
        poses_data.append(np.ma.concatenate([pose.body.data[:, :, :95],
                                             pose.body.data[:, :, 95:116],
                                             pose.body.data[:, :, 116:]], axis=2))
    return poses_data


def __compare_pred_to_video(pred, keypoints_path, pose_id, distance_function=fastdtw):
    label_pose = get_pose(os.path.join(keypoints_path, pose_id), pose_id)
    return compare_poses(pred, label_pose, distance_function=distance_function)


def check_ranks(distances, index):
    rank_1 = (index == distances[0])
    rank_5 = (index in distances[:5])
    rank_10 = (index in distances)
    return rank_1, rank_5, rank_10


def get_poses_ranks(pred, pred_id, keypoints_path, data_ids, distance_function=fastdtw, num_samples=20,
                    model=None, pose_header=None, ds=None):
    pred2label_distance = __compare_pred_to_video(pred, keypoints_path, pred_id, distance_function=distance_function)

    distances_to_label = [pred2label_distance]
    distances_to_pred = [pred2label_distance]
    pred2label_index = 0

    if model is not None:
        indices = random.sample(range(len(ds)), num_samples)
        for idx in indices:
            if ds[idx]["id"] == pred_id:
                continue
            cur_pred = predict_pose(model, ds[idx], pose_header)
            distances_to_label.append(__compare_pred_to_video(cur_pred, keypoints_path, pred_id,
                                                              distance_function=distance_function))
            distances_to_pred.append(compare_poses(pred, cur_pred, distance_function=distance_function))

    pose_ids = random.sample(data_ids, num_samples)
    for pose_id in pose_ids:
        distances_to_label.append(compare_pose_videos(pose_id, pred_id, keypoints_path,
                                                      distance_function=distance_function))
        distances_to_pred.append(__compare_pred_to_video(pred, keypoints_path, pose_id,
                                                         distance_function=distance_function))

    best_pred = np.argsort(distances_to_pred)[:10]
    rank_1_pred, rank_5_pred, rank_10_pred = check_ranks(best_pred, pred2label_index)
    best_label = np.argsort(distances_to_label)[:10]
    rank_1_label, rank_5_label, rank_10_label = check_ranks(best_label, pred2label_index)

    return pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, rank_10_label
