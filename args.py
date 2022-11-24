import random
from argparse import ArgumentParser
from os import path
import numpy as np
import torch

root_dir = path.dirname(path.realpath(__file__))
parser = ArgumentParser()

parser.add_argument('--no_wandb', type=bool, default=False, help='ignore wandb?')
parser.add_argument('--config_file', type=str, default="", help='path to yaml config file')

# Training Arguments
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_gpus', type=int, default=1, help='how many gpus?')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--max_epochs', type=int, default=2000, help='max number of epochs')
parser.add_argument('--masked_loss', type=bool, default=True, help='mask loss by confidence?')
parser.add_argument('--tf_p', type=float, default=0.5, help='percentage of teacher_forcing during training')
parser.add_argument('--optimizer', type=str, default='Adam', help='which optimizer to use? currently only Adam, '
                                                            'SGD are supported')

# Data Arguments
parser.add_argument('--max_seq_size', type=int, default=200, help='input sequence size')
parser.add_argument('--fps', type=int, default=25, help='fps to load')
parser.add_argument('--pose', choices=['openpose', 'holistic'], default='openpose', help='which pose estimation model to use?')
parser.add_argument('--pose_components', type=list, default=None, help='what pose components to use?')
parser.add_argument('--dataset', type=str, default='hamnosys', help='name of dataset')
parser.add_argument('--leave_out', type=str, default='', help='leave out a language? for the hamnosys dataset, '
                                                              'options are: ["pjm", "dgs", "gsl", "lsf"]. Default- use '
                                                              'all languages.')

# Model Arguments
parser.add_argument('--model_name', type=str, default="ham2pose", help='name of the model')
parser.add_argument('--noise_epsilon', type=float, default=1e-4, help='noise epsilon')
parser.add_argument('--seq_len_weight', type=float, default=2e-5, help='sequence length weight in loss calculation')
parser.add_argument('--hidden_dim', type=int, default=128, help='encoder hidden dimension')
parser.add_argument('--text_encoder_depth', type=int, default=2, help='number of layers for the text encoder')
parser.add_argument('--pose_encoder_depth', type=int, default=4, help='number of layers for the pose encoder')
parser.add_argument('--encoder_heads', type=int, default=2, help='number of heads for the encoder')
parser.add_argument('--num_steps', type=int, default=10, help='number of pose refinement steps')
parser.add_argument('--separate_positional_embedding', type=bool, default=True, help='separate positional embeddings '
                                                                                     'between text and pose?')
parser.add_argument('--num_pose_projection_layers', type=int, default=1, help='number of pose projection layers')
parser.add_argument('--encoder_dim_feedforward', type=int, default=2048, help='size of encoder dim feedforward')

# Prediction args
parser.add_argument('--ckpt', type=str, default="", metavar='PATH', help="Checkpoint path for prediction")
parser.add_argument('--output_dir', type=str, default="", metavar='PATH', help="Path for saving prediction files")

# test args
parser.add_argument('--test', type=bool, default=False, help="test?")

args = parser.parse_args()

# ---------------------
# Set Seed
# ---------------------
if args.seed == 0:  # Make seed random if 0
    args.seed = random.randint(0, 1000)
args.seed = 42
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
