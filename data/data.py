from typing import List, Dict
import torch
from pose_format import Pose
from torch.utils.data import Dataset
from data.tfds_dataset import ProcessedPoseDatum, get_tfds_dataset
from constants import MIN_CONFIDENCE, NUM_FACE_KEYPOINTS


class TextPoseDatum(Dict):
    id: str
    text: str
    pose: Pose
    length: int


class TextPoseDataset(Dataset):
    def __init__(self, data: List[TextPoseDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        if "pose" not in datum:
            return datum

        pose = datum["pose"]
        torch_body = pose.body.torch()
        pose_length = len(torch_body.data)

        return {
            "id": datum["id"],
            "text": datum["text"],
            "pose": {
                "obj": pose,
                "data": torch_body.data.tensor[:, 0, :, :],
                "confidence": torch_body.confidence[:, 0, :],
                "length": torch.tensor([pose_length], dtype=torch.float),
                "inverse_mask": torch.ones(pose_length, dtype=torch.bool)
            }
        }


def process_datum(datum: ProcessedPoseDatum) -> TextPoseDatum:
    text = datum["tf_datum"]["hamnosys"].numpy().decode('utf-8').strip() \
        if "hamnosys" in datum["tf_datum"] else ""
    text = str(datum["tf_datum"]["gloss_id"].numpy()) if "gloss_id" in datum["tf_datum"] else text
    if "pose" not in datum:
        return TextPoseDatum({
            "id": datum["id"],
            "text": text,
            "pose_len": float(datum["tf_datum"]["pose_len"].numpy()),
            "length": max(datum["tf_datum"]["pose_len"], len(text))
        })

    pose: Pose = datum["pose"]

    face_th = 0.5*NUM_FACE_KEYPOINTS
    hands_th = MIN_CONFIDENCE

    # Prune all leading frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i][:, 25:-42].sum() > face_th and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > hands_th:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Prune all trailing frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i][:, 25:-42].sum() > face_th and \
                pose.body.confidence[i][:, 4] + pose.body.confidence[i][:, 7] > hands_th:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return TextPoseDatum({
        "id": datum["id"],
        "text": text,
        "pose": pose,
        "length": len(pose.body.data)
    })


def get_dataset(name="dicta_sign", poses="openpose", fps=25, split="train", components: List[str] = None,
                data_dir=None, max_seq_size=200, leave_out=""):

    data = get_tfds_dataset(name=name, poses=poses, fps=fps, split=split, components=components,
                            data_dir=data_dir)

    data = [process_datum(d) for d in data]
    data = [d for d in data if d["length"] < max_seq_size]
    if leave_out != "":
        if leave_out == "dgs":
            train_data = [d for d in data if not d["id"].isnumeric()]
            test_data = [d for d in data if d["id"].isnumeric()]
        elif leave_out == "lsf":
            train_data = [d for d in data if any(i.isdigit() for i in d["id"])]
            test_data = [d for d in data if not any(i.isdigit() for i in d["id"])]
        else:
            train_data = [d for d in data if leave_out not in d["id"]]
            test_data = [d for d in data if leave_out in d["id"]]
        return TextPoseDataset(train_data), TextPoseDataset(test_data)
    return TextPoseDataset(data)
