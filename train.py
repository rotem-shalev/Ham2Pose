import os
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import sys

rootdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, rootdir)

from data.collator import zero_pad_collator
from args import args
from data.data import get_dataset
from model import IterativeTextGuidedPoseGenerationModel
from tokenizers.hamnosys.hamnosys_tokenizer import HamNoSysTokenizer
from predict import pred
from constants import num_steps_to_batch_size, batch_size_to_accumulate, DATASET_SIZE


def get_optimizer(opt_str):
    if opt_str == "Adam":
        return Adam
    elif opt_str == "SGD":
        return SGD
    else:
        raise Exception("optimizer not supported. use Adam or SGD.")


def get_model_args(args, num_pose_joints, num_pose_dims):
    model_args = dict(tokenizer=HamNoSysTokenizer(),
                      pose_dims=(num_pose_joints, num_pose_dims),
                      hidden_dim=args["hidden_dim"],
                      text_encoder_depth=args["text_encoder_depth"],
                      pose_encoder_depth=args["pose_encoder_depth"],
                      encoder_heads=args["encoder_heads"],
                      max_seq_size=args["max_seq_size"],
                      num_steps=args["num_steps"],
                      tf_p=args["tf_p"],
                      seq_len_weight=args["seq_len_weight"],
                      noise_epsilon=args["noise_epsilon"],
                      optimizer_fn=get_optimizer(args["optimizer"]),
                      separate_positional_embedding=args["separate_positional_embedding"],
                      encoder_dim_feedforward=args["encoder_dim_feedforward"],
                      num_pose_projection_layers=args["num_pose_projection_layers"]
                      )

    return model_args


if __name__ == '__main__':
    args = vars(args)
    if args["config_file"]:  # override args with yaml config file
        with open(args["config_file"], 'r') as f:
            args = yaml.safe_load(f)

    LOGGER = None
    if not args["no_wandb"]:
        LOGGER = WandbLogger(project="ham2pose", log_model=False, offline=False, id=args["model_name"])
        if LOGGER.experiment.sweep_id is None:
            LOGGER.log_hyperparams(args)
    args["batch_size"] = num_steps_to_batch_size[args["num_steps"]]

    test_size = int(0.1*DATASET_SIZE)
    train_split = f'test[{test_size}:]+train'
    test_split = f'test[:{test_size}]'

    if args["leave_out"] != "":
        train_dataset, test_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
                                    components=args["pose_components"], leave_out=args["leave_out"],
                                    max_seq_size=args["max_seq_size"], split=train_split)
    else:
        train_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
                                   components=args["pose_components"], max_seq_size=args["max_seq_size"],
                                    split=train_split)
        test_dataset = get_dataset(name=args["dataset"], poses=args["pose"], fps=args["fps"],
                                   components=args["pose_components"], max_seq_size=args["max_seq_size"],
                                   split=test_split)

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"],
                              shuffle=True, collate_fn=zero_pad_collator)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"],
                                   collate_fn=zero_pad_collator)

    _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    model_args = get_model_args(args, num_pose_joints, num_pose_dims)

    if os.path.isfile(f"./models/{args['model_name']}/{args['ckpt']}/model.ckpt"):
        model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(f"./models/{args['model_name']}/"
                                                                            f"{args['ckpt']}/model.ckpt", **model_args)
    else:
        model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = []
    if LOGGER is not None:
        os.makedirs(f"./models/{args['model_name']}", exist_ok=True)
        callbacks.append(ModelCheckpoint(
            dirpath=f"./models/{args['model_name']}",
            filename="model",
            verbose=True,
            save_top_k=1,
            monitor='train_loss',
            mode='min'
        ))

    trainer = pl.Trainer(
        max_epochs=args['max_epochs'],
        logger=LOGGER,
        callbacks=callbacks,
        accelerator='gpu',
        devices=args['num_gpus'],
        accumulate_grad_batches=batch_size_to_accumulate[args['batch_size']],
        strategy="ddp"
    )

    trainer.fit(model, train_dataloaders=train_loader)

    # evaluate
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(f"./models/{args['model_name']}/"
                                                                        f"{args['ckpt']}/model.ckpt", **model_args)
    model.eval()

    # test seq_len_predictor
    diffs = []
    for d in test_dataset:
        _, seq_len = model.encode_text([d["text"]])
        real_seq_len = len(d["pose"]["data"])
        diff = np.abs(real_seq_len-seq_len.item())
        diffs.append(diff)
    print(np.mean(diffs), np.median(diffs), np.max(diffs))

    pred(model, train_dataset, os.path.join(f"./models/{args['model_name']}", args['output_dir'], "train"))
    pred(model, test_dataset, os.path.join(f"./models/{args['model_name']}", args['output_dir'], "test"))
