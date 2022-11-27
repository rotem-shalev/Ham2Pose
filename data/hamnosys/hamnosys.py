"""hamnosys dataset."""

import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from os import path
import json
import zipfile
from typing import Dict
from pose_format.utils.openpose import load_openpose
from pose_format.pose import Pose

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), ".."))
# sys.path.append(os.path.join(os.path.dirname(SCRIPT_DIR), "..", "utils"))

from data.config import SignDatasetConfig  # for compilation remove ".."
from data.utils.features import PoseFeature  # for compilation remove ".."

_DESCRIPTION = """
Combined corpus of videos with their openPose keypoints and HamNoSys. 
Includes dicta_sign, KSPJM (The Corpus Dictionary of Polish Sign Language).
"""

_CITATION = """
@inproceedings{efthimiou2010dicta,
  title={Dicta-sign--sign language recognition, generation and modelling: a research effort with applications in deaf communication},
  author={Efthimiou, Eleni and Fontinea, Stavroula-Evita and Hanke, Thomas and Glauert, John and Bowden, Rihard and Braffort, Annelies and Collet, Christophe and Maragos, Petros and Goudenove, Fran{\c{c}}ois},
  booktitle={Proceedings of the 4th Workshop on the Representation and Processing of Sign Languages: Corpora and Sign Language Technologies},
  pages={80--83},
  year={2010}
}

@inproceedings{linde2014corpus,
  title={A corpus-based dictionary of Polish Sign Language (PJM)},
  author={Linde-Usiekniewicz, Jadwiga and Czajkowska-Kisil, Ma{\l}gorzata and {\L}acheta, Joanna and Rutkowski, Pawe{\l}},
  booktitle={Proceedings of the XVI EURALEX International Congress: The user in focus},
  pages={365--376},
  year={2014}
}
"""

MAX_HEIGHT = 400
MAX_WIDTH = 400
MIN_CONFIDENCE = 0.2

# signs with HamNoSys that appears more than once- must be in train!
DUP_KEYS = ['10248', '3516', '44909', '10573', '12916', '2674', '10753', '8044', '10890', '69225', '9280',
            '11286', '48575', '68699', '11288', '27428', '6248', '11291', '75271', '11420', '39949', '11435',
            '59785', '6230', '11874', '2294', '12278', '3071', '12641', '59684', '12844', '59701', '15121', '85192',
            '15286', '59212', '15735', '20652', '15962', '2803', '16153', '40233', '17265', '67630', '18003', '89436',
            '2442', '3048', '9028', '2452', '2856', '25235', '4511', '2686', '5035', '27521', '87394', '29817', '86689',
            '30365', '4171', '3172', '40005', '5908', '3193', '88457', '43516', '65542', '48749', '68018', '53036',
            '9386', '5492', '91376', '55848', '72736', '56000', '76667', '56684', '58318', '59424', '6192', '60848',
            '73060', '61731', '7247', '8291', '71120', '85160', '76557', '80774', '7940', '9790', '8265', '87255',
            '8289', '87848', 'FEUILLE', 'PAPIER', 'gsl_1024', 'gsl_165', 'gsl_124', 'gsl_51', 'gsl_145', 'gsl_804',
            'gsl_148', 'gsl_212', 'gsl_189', 'gsl_318', 'gsl_236', 'gsl_585', 'gsl_244', 'gsl_965', 'gsl_27', 'gsl_504',
            'gsl_339', 'gsl_530', 'gsl_353', 'gsl_719', 'gsl_424', 'gsl_923', 'gsl_475', 'gsl_545', 'gsl_495', 'gsl_883',
            'gsl_528', 'gsl_692']

_POSE_HEADERS = {
    "openpose": path.join(path.dirname(path.realpath(__file__)), "openpose.poseheader"),
}
_DATA_PATH = path.join(path.dirname(path.realpath(__file__)), "data.json")
_KEYPOINTS_PATH = path.join(path.dirname(path.realpath(__file__)), "keypoints")


def get_pose(keypoints_path: str, fps: int = 25) -> Dict[str, Pose]:
    """
    Load OpenPose in the particular format (a single file for all frames).
    :param keypoints_path: Path to a folder that contains keypoints jsons (OpenPose output)
    for all frames of a video.
    :param fps: frame rate. default is 25.
    :return: Dictionary of Pose object with a header specific to OpenPose and a body that contains a
    single array.
    """
    files = sorted(tf.io.gfile.listdir(keypoints_path))
    frames = dict()
    for i, file in enumerate(files):
        try:
            with tf.io.gfile.GFile(path.join(keypoints_path, file), "r") as openpose_raw:
                frame_json = json.load(openpose_raw)
                frames[i] = {"people": frame_json["people"][:1], "frame_id": i}
                cur_frame_pose = frame_json["people"][0]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]) -
                    np.array(cur_frame_pose['hand_left_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][7*3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_left_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][7*3:7*3 + 2]
                if (np.array(cur_frame_pose['pose_keypoints_2d'][4*3:4*3 + 2]) -
                    np.array(cur_frame_pose['hand_right_keypoints_2d'][0:2])).max() > 15 and \
                        cur_frame_pose['pose_keypoints_2d'][4*3 + 2] > MIN_CONFIDENCE:
                    cur_frame_pose['hand_right_keypoints_2d'][0:2] = cur_frame_pose['pose_keypoints_2d'][4*3:4*3 + 2]

        except:
            continue

    if len(frames) == 0:
        print(keypoints_path)
        return None

    # Convert to pose format
    pose = load_openpose(frames, fps=fps, width=400, height=400)
    return pose


class Hamnosys(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for hamnosys dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    BUILDER_CONFIGS = [
        SignDatasetConfig(name="default", include_video=False, include_pose="openpose", fps=25),
        SignDatasetConfig(name="videos", include_video=True, include_pose="openpose"),
        SignDatasetConfig(name="text_only", include_video=False, include_pose=None, fps=0),
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        if self._builder_config.include_pose is None:
            features = tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "hamnosys": tfds.features.Text(),
                "text": tfds.features.Text(),
                "pose_len": tf.float32,
            })
        else:
            pose_header_path = _POSE_HEADERS[self._builder_config.include_pose]
            stride = 1  # if self._builder_config.fps is None else 50 / self._builder_config.fps
            pose_shape = (None, 1, 137, 2)
            features = tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "video": tfds.features.Text(),  # tfds.features.Video(shape=(None, MAX_HEIGHT, MAX_WIDTH, 3)),
                "fps": tf.int32,
                "pose": PoseFeature(shape=pose_shape, stride=stride, header_path=pose_header_path),
                "hamnosys": tfds.features.Text(),
                "text": tfds.features.Text(),
            })

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=features,
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits
        with tf.io.gfile.GFile(_DATA_PATH) as f:
            data = json.load(f)
        return [tfds.core.SplitGenerator(name=tfds.Split.TRAIN, gen_kwargs={"data": data, "is_train": True}),
                tfds.core.SplitGenerator(name=tfds.Split.TEST, gen_kwargs={"data": data, "is_train": False})]

    def _generate_examples(self, data, is_train):
        """Yields examples."""
        if not os.path.isdir(_KEYPOINTS_PATH) and os.path.isfile("data/hamnosys/keypoints.zip"):
            with zipfile.ZipFile("data/hamnosys/keypoints.zip", 'r') as zip_ref:
                zip_ref.extractall(_KEYPOINTS_PATH)
        default_fps = 25
        i = 0
        for key, val in data.items():
            if not tf.io.gfile.isdir(path.join(_KEYPOINTS_PATH, key)):
                continue
            if (is_train and key not in DUP_KEYS) or (not is_train and key in DUP_KEYS):
                continue

            features = {
                "id": key,
                "hamnosys": val["hamnosys"],
                "text": val["type_name"]
            }

            if self._builder_config.include_pose is not None:
                features["video"] = val["video_frontal"]
                features["fps"] = default_fps
                features["pose"] = get_pose(path.join(_KEYPOINTS_PATH, key), fps=default_fps)
            else:
                features["pose_len"] = len(tf.io.gfile.listdir(path.join(_KEYPOINTS_PATH, key)))

            i += 1
            yield key, features
