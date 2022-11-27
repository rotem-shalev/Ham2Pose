import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from args import args
from data.data import get_dataset
from constants import DATASET_SIZE, num_steps_to_batch_size
from model import IterativeTextGuidedPoseGenerationModel
from train import get_model_args
from predict import pred, predict_pose
from metrics import get_poses_ranks


def combine_results(experiment_name, results_path):
    results = dict()
    for file in os.listdir(results_path):
        if experiment_name in file:
            with open(os.path.join(results_path, file)) as f:
                results.update(json.load(f))
    return np.mean(list(results.values())), np.median(list(results.values()))


def get_lang(sign_id):
    if "pjm" in sign_id:
        return "pjm"
    elif "gsl" in sign_id:
        return "gsl"
    elif sign_id.isnumeric():
        return "dgs"
    else:
        return "lsf"


def get_results_by_language(filename, num_files=5):
    paths = [os.path.join(filename+f"_{i}.txt") for i in range(num_files)]
    languages = {"pjm", "dgs", "gsl", "lsf"}
    all_results = {lang: {"pred_rank1": 0, "pred_rank5": 0, "pred_rank10": 0, "gt_rank1": 0, "gt_rank5": 0,
                       "gt_rank10": 0} for lang in languages}
    lang2count = {lang: 0 for lang in languages}
    for path in paths:
        with open(path, 'r') as f:
            lines = f.readlines()
            for i in range(1, len(lines), 2):
                if lines[i].startswith("rank"):
                    break
                lang = get_lang(lines[i].split(" ")[0])
                lang2count[lang] += 1
                dist, pred_rank1, pred_rank5, pred_rank10, gt_rank1, gt_rank5, gt_rank10 = lines[i+1].strip().split(", ")
                all_results[lang]["pred_rank1"] += int(pred_rank1 == "True")
                all_results[lang]["pred_rank5"] += int(pred_rank5 == "True")
                all_results[lang]["pred_rank10"] += int(pred_rank10 == "True")
                all_results[lang]["gt_rank1"] += int(gt_rank1 == "True")
                all_results[lang]["gt_rank5"] += int(gt_rank5 == "True")
                all_results[lang]["gt_rank10"] += int(gt_rank10 == "True")

    for lang, ranks in all_results.items():
        for rank, rank_count in ranks.items():
            rank_mean = rank_count / lang2count[lang]
            print(f"{rank} of {lang} is: {rank_count}/{lang2count[lang]}= {rank_mean}")


def test_seq_len(model, dataset, model_name):
    abs_diffs = dict()
    diffs = dict()
    for d in dataset:
        _, seq_len = model.encode_text([d["text"]])
        real_seq_len = d["pose"]["length"]
        diff = seq_len.item() - real_seq_len.item()
        abs_diffs[d["id"]] = np.abs(diff)
        diffs[d["id"]] = diff / real_seq_len.item()
    print(f"mean diff: {np.mean(list(diffs.values()))}, median: {np.median(list(diffs.values()))}")
    print(f"mean absolute diff: {np.mean(list(abs_diffs.values()))}, median:"
          f" {np.median(list(abs_diffs.values()))}")

    plt.hist([v * 100 for v in diffs.values()], bins=80)
    plt.xticks(ticks=[-50, 0, 50, 100, 150], labels=["-50%", "0%", "50%", "100%", "150%"])
    # plt.title("Predicted vs. real sequence length difference")
    plt.xlabel('sequence length error percentage')
    plt.ylabel('Count')
    plt.savefig(f"models/{model_name}/results/seq_len_diff_hist.png")
    plt.clf()

    plt.hist(list(abs_diffs.values()), bins=10)
    # plt.title("Predicted vs. real sequence length absolute difference")
    plt.xlabel('frame number difference (FPS=25)')
    plt.ylabel('Count')
    plt.savefig(f"models/{model_name}/results/seq_len_abs_diff_hist.png")
    plt.clf()

    with open(f"models/{model_name}/results/seq_len_diffs.json", 'w') as f:
        json.dump(diffs, f)
    with open(f"models/{model_name}/results/seq_len_abs_diffs.json", 'w') as f:
        json.dump(abs_diffs, f)


def test_distance_ranks(model, model_name, dataset, keypoints_path, num_samples=20):
    keypoints_dirs = os.listdir(keypoints_path)
    with open("data/hamnosys/data.json", 'r') as f:
        data = json.load(f)
        data_ids = list(filter(lambda x: x in keypoints_dirs, data.keys()))

    model = model.cuda()
    with torch.no_grad():
        rank_1_pred_sum = rank_5_pred_sum = rank_10_pred_sum = rank_1_label_sum = rank_5_label_sum = \
            rank_10_label_sum = 0
        pred2label_distances = dict()
        for datum in dataset:
            if len(datum["pose"]["data"]) == 0:
                continue
            predicted_pose = predict_pose(model, datum, pose_header)
            pred2label_distance, rank_1_pred, rank_5_pred, rank_10_pred, rank_1_label, rank_5_label, \
            rank_10_label = get_poses_ranks(predicted_pose, datum["id"], keypoints_path, data_ids,
                                            model=model, pose_header=pose_header, ds=dataset, num_samples=num_samples)
            pred2label_distances[datum["id"]] = pred2label_distance
            print(f"{datum['id']} ranks:\n "
                  f"{pred2label_distance}, {rank_1_pred}, {rank_5_pred}, {rank_10_pred}, {rank_1_label},"
                  f" {rank_5_label}, {rank_10_label}")
            rank_1_pred_sum += int(rank_1_pred)
            rank_5_pred_sum += int(rank_5_pred)
            rank_10_pred_sum += int(rank_10_pred)
            rank_1_label_sum += int(rank_1_label)
            rank_5_label_sum += int(rank_5_label)
            rank_10_label_sum += int(rank_10_label)

        num_samples = len(dataset)
        print(f"rank 1 pred sum: {rank_1_pred_sum} / {num_samples}: {rank_1_pred_sum / num_samples}")
        print(f"rank 5 pred sum: {rank_5_pred_sum} / {num_samples}: {rank_5_pred_sum / num_samples}")
        print(f"rank 10 pred sum: {rank_10_pred_sum} / {num_samples}: {rank_10_pred_sum / num_samples}")

        print(f"rank 1 label sum: {rank_1_label_sum} / {num_samples}: {rank_1_label_sum / num_samples}")
        print(f"rank 5 label sum: {rank_5_label_sum} / {num_samples}: {rank_5_label_sum / num_samples}")
        print(f"rank 10 label sum: {rank_10_label_sum} / {num_samples}: {rank_10_label_sum / num_samples}")

        with open(f"models/{model_name}/results/pred2label_distances_NDTW_pred_label_gallery.json", 'w') as f:
            json.dump(pred2label_distances, f)

        print(f"mean distance between pred and label: {np.mean(list(pred2label_distances.values()))}")
        print(f"median distance between pred and label: {np.median(list(pred2label_distances.values()))}")

        plt.hist(list(pred2label_distances.values()))
        plt.title("DTW distance between ground truth and predicted pose")
        plt.savefig(f"models/{model_name}/results/pred2label_distances_hist.png")


def test(model, model_name, dataset, test_seq_len_predictor=True, test_ranks=True, output_dir="",
         keypoints_path=""):

    os.makedirs(f"models/{model_name}/results", exist_ok=True)

    if output_dir != "":
        pred(model, dataset, f"models/{model_name}/{output_dir}")

    if test_seq_len_predictor:
        test_seq_len(model, dataset, model_name)

    if test_ranks:
        test_distance_ranks(model, model_name, dataset, keypoints_path)


if __name__ == "__main__":
    args = vars(args)
    if args["config_file"]:  # override args with yaml config file
        with open(args["config_file"], 'r') as f:
            args = yaml.safe_load(f)

    args["batch_size"] = num_steps_to_batch_size[args["num_steps"]]
    test_size = int(0.1 * DATASET_SIZE)
    if args["leave_out"] != "":
        _, dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
                                 components=args["pose_components"], leave_out=args["leave_out"],
                                 max_seq_size=args["max_seq_size"], split='test')
    else:
        dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
                              components=args["pose_components"], max_seq_size=args["max_seq_size"],
                              split=f'test[:{test_size}]')

    _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset.data[0]["pose"].header

    model_args = get_model_args(args, num_pose_joints, num_pose_dims)

    ckpt = f"./models/{args['model_name']}/{args['ckpt']}/model.ckpt"
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(ckpt, **model_args)
    model.eval()

    test(model, args["model_name"], dataset, test_seq_len_predictor=True, test_ranks=False,
         output_dir=args["output_dir"], keypoints_path="data/hamnosys/keypoints")

