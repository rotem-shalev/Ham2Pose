import os
import numpy as np
import cv2
from typing import List
import torch
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_visualizer import PoseVisualizer

import sys
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from pose_utils import pose_normalization_info, pose_hide_legs


def pred(model, dataset, output_dir, vis_process=False, gen_k=30, vis=True, subset=None):
    os.makedirs(output_dir, exist_ok=True)
    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset[0]["pose"]["obj"].header
    preds = []

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if subset is not None and datum["id"] not in subset:
                continue
            if i >= gen_k and subset is None:
                break
            first_pose = datum["pose"]["data"][0]
            sequences = []
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
            for j in range(model.num_steps):
                seq = next(seq_iter)
                if vis_process and j in [0, 2, 3, 4, 9]:
                    sequences.append(seq)

            if vis_process:
                visualize_sequences(sequences, pose_header, output_dir, datum["id"], datum["pose"]["obj"])
            elif vis:
                visualize_seq(seq, pose_header, output_dir, datum["id"], datum["pose"]["obj"])
            else:
                data = torch.unsqueeze(seq, 1).cpu()
                conf = torch.ones_like(data[:, :, :, 0])
                pose_body = NumPyPoseBody(25, data.numpy(), conf.numpy())
                predicted_pose = Pose(pose_header, pose_body)
                pose_hide_legs(predicted_pose)
                preds.append(predicted_pose)
    return preds


def visualize_seq(seq, pose_header, output_dir, id, label_pose=None, fps=25):
    with torch.no_grad():
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        predicted_pose = Pose(pose_header, pose_body)
        if "pose_keypoints_2d" in [c.name for c in pose_header.components]:
            pose_hide_legs(predicted_pose)

        pose_name = f"{id}.mp4"
        if label_pose is None:
            visualize_pose([predicted_pose], pose_name, output_dir)
        else:
            visualize_pose([label_pose, predicted_pose], pose_name, output_dir)


def visualize_sequences(sequences, pose_header, output_dir, id, label_pose=None, fps=25, labels=None):
    os.makedirs(output_dir, exist_ok=True)
    poses = [label_pose]
    for seq in sequences:
        data = torch.unsqueeze(seq, 1).cpu()
        conf = torch.ones_like(data[:, :, :, 0])
        pose_body = NumPyPoseBody(fps, data.numpy(), conf.numpy())
        pose = Pose(pose_header, pose_body)
        pose_hide_legs(pose)
        poses.append(pose)

    pose_name = f"{id}_merged.mp4"
    font = cv2.FONT_HERSHEY_TRIPLEX
    color = (0, 0, 0)
    f_name = os.path.join(output_dir, pose_name)
    all_frames = get_normalized_frames(poses)
    if labels is None:
        labels = ["ground truth", "step 10", "step 8", "step 7", "step 6", "last step"]
    text_margin = 50
    w = max([frames[0].shape[1] for frames in all_frames])
    h = max([frames[0].shape[0] for frames in all_frames])
    image_size = (w * len(poses), h + text_margin)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    max_len = max([len(frames) for frames in all_frames])

    for i in range(max_len+1):
        all_video_frames = []
        for j, frames in enumerate(all_frames):
            if len(frames) > i:
                cur_frame = np.full((image_size[1], image_size[0] // len(poses), 3), 255, dtype=np.uint8)
                cur_frame[text_margin:frames[i].shape[0] + text_margin, :frames[i].shape[1]] = frames[i]
                cur_frame = cv2.putText(cur_frame, labels[j], (5, 20), font, 0.5, color, 1, 0)
            else:
                cur_frame = prev_video_frames[j]
            all_video_frames.append(cur_frame)
        merged_frame = np.concatenate(all_video_frames, axis=1)
        out.write(merged_frame)
        prev_video_frames = all_video_frames

    out.release()


def visualize_pose(poses: List[Pose], pose_name: str, output_dir: str, slow: bool = False):
    os.makedirs(output_dir, exist_ok=True)
    f_name = os.path.join(output_dir, pose_name)
    fps = poses[0].body.fps
    if slow:
        f_name = f_name[:-4] + "_slow" + ".mp4"
        fps = poses[0].body.fps // 2

    frames = get_normalized_frames(poses)

    if len(poses) == 1:
        image_size = (frames[0][0].shape[1], frames[0][0].shape[0])
        out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
        for frame in frames[0]:
            out.write(frame)
        out.release()
        return

    text_margin = 50
    image_size = (max(frames[0][0].shape[1], frames[1][0].shape[1]) * 2,
                  max(frames[0][0].shape[0], frames[1][0].shape[0]) + text_margin)
    out = cv2.VideoWriter(f_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, image_size)
    for i, frame in enumerate(frames[1]):
        if len(frames[0]) > i:
            empty = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
            empty[text_margin:frames[0][i].shape[0]+text_margin, :frames[0][i].shape[1]] = frames[0][i]
        label_frame = empty
        pred_frame = np.full((image_size[1], image_size[0]//2, 3), 255, dtype=np.uint8)
        pred_frame[text_margin:frame.shape[0]+text_margin, :frame.shape[1]] = frame
        label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
        out.write(label_pred_im)

    if i < len(frames[0])-1:
        for frame in frames[0][i:]:
            label_frame = np.full((image_size[1], image_size[0] // 2, 3), 255, dtype=np.uint8)
            label_frame[text_margin:frame.shape[0] + text_margin, :frame.shape[1]] = frame
            label_pred_im = concat_and_add_label(label_frame, pred_frame, image_size)
            out.write(label_pred_im)

    out.release()


def concat_and_add_label(label_frame, pred_frame, image_size):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 250, 0)
    label_pred_im = np.concatenate((label_frame, pred_frame), axis=1)
    label_pred_im = cv2.putText(label_pred_im, "label", (image_size[0] // 5, 30), font, 1,
                                color, 2, 0)
    label_pred_im = cv2.putText(label_pred_im, "pred",
                                (image_size[0] // 5 + image_size[0] // 2, 30),
                                font, 1, color, 2, 0)
    return label_pred_im


def get_normalized_frames(poses):
    frames = []
    for i in range(len(poses)):
        # Normalize pose
        normalization_info = pose_normalization_info(poses[i].header)
        pose = poses[i].normalize(normalization_info, scale_factor=100)
        pose.focus()
        visualizer = PoseVisualizer(pose, thickness=2)
        pose_frames = list(visualizer.draw())
        frames.append(pose_frames)

    return frames


def predict_pose(model, datum, pose_header, model_for_seq_len=None, first_pose_seq=False):
    first_pose = datum["pose"]["data"][0]
    if model_for_seq_len is not None:
        seq_len = int(model_for_seq_len.encode_text([datum["text"]])[1].item())
    elif first_pose_seq:
        seq_len = int(model.encode_text([datum["text"]])[1].item())
    else:
        seq_len = -1

    if first_pose_seq:
        seq = torch.stack([first_pose] * seq_len, dim=0)
    else:
        seq_iter = model.forward(text=datum["text"], first_pose=first_pose.cuda(), sequence_length=seq_len)
        for j in range(model.num_steps):
            seq = next(seq_iter)

    data = torch.unsqueeze(seq, 1).cpu()
    conf = torch.ones_like(data[:, :, :, 0])
    pose_body = NumPyPoseBody(25, data.detach().numpy(), conf.numpy())
    predicted_pose = Pose(pose_header, pose_body)
    pose_hide_legs(predicted_pose)
    normalization_info = pose_normalization_info(predicted_pose.header)
    predicted_pose = predicted_pose.normalize(normalization_info)
    predicted_pose.focus()
    return predicted_pose
