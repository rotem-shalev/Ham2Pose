import numpy as np
from numpy import ma
from collections import OrderedDict
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions, PoseHeader
from pose_format.utils.openpose import OpenPose_Components


def pose_hide_low_conf(pose: Pose):
    mask = pose.body.confidence <= 0.2
    pose.body.confidence[mask] = 0
    stacked_confidence = np.stack([mask, mask], axis=3)
    masked_data = ma.masked_array(pose.body.data, mask=stacked_confidence)
    pose.body.data = masked_data


def pose_hide_legs(pose: Pose):
    if pose.header.components[0].name in ["pose_keypoints_2d", 'BODY_135']:
        point_names = ["Knee", "Ankle", "Heel", "BigToe", "SmallToe", "Hip"]
        points = [pose.header._get_point_index(pose.header.components[0].name, side+n)
                for n in point_names for side in ["L", "R"]]
    elif pose.header.components[0].name == "POSE_LANDMARKS":
        point_names = ["KNEE", "ANKLE", "HEEL", "FOOT_INDEX"]
        # pylint: disable=protected-access
        points = [pose.header._get_point_index("POSE_LANDMARKS", side + "_" + n)
                  for n in point_names for side in ["LEFT", "RIGHT"]]
    else:
        raise ValueError("Unknown pose header schema for hiding legs")

    pose.body.confidence[:, :, points] = 0
    pose.body.data[:, :, points, :] = 0


def pose_normalization_info(pose_header: PoseHeader):
    if pose_header.components[0].name == "POSE_LANDMARKS":
        return pose_header.normalization_info(
            p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
            p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
        )

    if pose_header.components[0].name == "BODY_135":
        return pose_header.normalization_info(
            p1=("BODY_135", "RShoulder"),
            p2=("BODY_135", "LShoulder")
        )

    if pose_header.components[0].name == "pose_keypoints_2d":
        return pose_header.normalization_info(
            p1=("pose_keypoints_2d", "RShoulder"),
            p2=("pose_keypoints_2d", "LShoulder")
        )

    if pose_header.components[0].name == "hand_left_keypoints_2d":
        return pose_header.normalization_info(
            p1=("hand_left_keypoints_2d", "BASE"),
            p2=("hand_right_keypoints_2d", "BASE")
        )

    raise ValueError("Unknown pose header schema for normalization")


def get_node2parent_dict(only_hands=False):
    hand_node2parent = OrderedDict([(1, 0), (5, 0), (9, 0), (13, 0), (17, 0), (2, 1), (6, 5), (10, 9), (14, 13),
                                    (18, 17), (3, 2), (7, 6), (11, 10), (15, 14), (19, 18), (4, 3), (8, 7), (12, 11),
                                    (16, 15), (20, 19)])
    if only_hands:
        node2parent = OrderedDict([(0, 0)])
        node2parent.update(hand_node2parent)
        return node2parent
    node2parent = OrderedDict([(0, 0), (1, 0), (2, 1), (5, 1), (6, 5), (7, 6), (3, 2), (4, 3), (8, 1),
                               (16, 0), (15, 0), (18, 16), (17, 15)])
    face_node2parent = OrderedDict([(32, 33), (31, 32), (34, 33), (35, 34), (30, 33), (29, 30), (28, 29), (27, 28),
                                    (42, 27), (43, 42), (44, 43), (45, 44), (46, 45), (47, 46), (39, 27), (38, 39),
                                    (37, 38), (36, 37), (41, 36), (40, 41), (68, 36), (69, 42), (22, 27), (21, 27),
                                    (20, 21), (19, 20), (18, 19), (17, 18), (23, 22), (24, 23), (25, 24), (26, 25),
                                    (0, 17), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6), (8, 7), (9, 8),
                                    (10, 9), (11, 10), (12, 11), (13, 12), (14, 13), (15, 14), (16, 15), (60, 48),
                                    (61, 60), (62, 61), (63, 62), (64, 63), (65, 64), (66, 65), (67, 66), (51, 33),
                                    (50, 51), (52, 51), (53, 52), (54, 53), (55, 54), (56, 55), (57, 56), (58, 57),
                                    (59, 58), (49, 50), (48, 49)])
    lhand_root = 7
    rhand_root = 4

    FACE_INDEX = 25
    LHAND_INDEX = FACE_INDEX + 70
    RHAND_INDEX = LHAND_INDEX + 21

    # face root is body root = nose (33)
    node2parent.update([(FACE_INDEX + 33, 0), (LHAND_INDEX, lhand_root), (RHAND_INDEX, rhand_root)])
    node2parent.update(((key + FACE_INDEX, val + FACE_INDEX) for key, val in face_node2parent.items()))
    node2parent.update(((key + LHAND_INDEX, val + LHAND_INDEX) for key, val in hand_node2parent.items()))
    node2parent.update(((key + RHAND_INDEX, val + RHAND_INDEX) for key, val in hand_node2parent.items()))

    return node2parent


def fake_pose(num_frames: int, fps: int = 25):
    dimensions = PoseHeaderDimensions(width=1, height=1, depth=1)
    header = PoseHeader(version=0.1, dimensions=dimensions, components=OpenPose_Components)

    total_points = header.total_points()
    data = np.zeros(shape=(num_frames, 1, total_points, 2), dtype=np.float32)
    confidence = np.zeros(shape=(num_frames, 1, total_points), dtype=np.float32)
    masked_data = ma.masked_array(data)

    body = NumPyPoseBody(fps=int(fps), data=masked_data, confidence=confidence)

    return Pose(header, body)
