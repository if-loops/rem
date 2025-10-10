# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# base from https://github.com/drimpossible/corrective-unlearning-bench
"""Parses command-line arguments for the unlearning benchmark.

This module defines the command-line arguments used to configure datasets,
models, unlearning methods, and training parameters for the experiments.
"""
import argparse
import torch


def parse_args():
  """Parses command-line arguments for the unlearning benchmark.

  Returns:
    An argparse.Namespace containing the parsed arguments.
  """
  # Main Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--dataset",
      type=str,
      default="tinyimagenet",
      choices=[
          "CIFAR10",
          "CIFAR100",
          "tinyimagenet",
          "Imagenette",
          "PCAM",
          "SVHN",
          "LFWPeople",
          "CelebA",
          "DermNet",
          "Pneumonia",
      ],
  )
  parser.add_argument(
      "--model",
      type=str,
      default="resnet9",
      choices=[
          "resnet9",
          "resnet20",
          "resnet32",
          "resnet44",
          "resnet56",
          "resnet110",
          "resnetwide28x10",
          "vitb16",
          "ViT",
      ],
  )
  parser.add_argument(
      "--dataset_method",
      type=str,
      default="labelrandom",
      choices=["randomlabelswap", "interclasslabelswap", "poisoning"],
      help="Number of Classes",
  )
  parser.add_argument(
      "--unlearn_method",
      type=str,
      default="Naive",
      choices=[
          "Naive",
          "EU",
          "SalUn",
          "CF",
          "REMMOD",
          "Scrub",
          "NPOREMMOD",
          "REMIDEAL",
          "REMMODIDEAL",
          "NPO",
          "REM",
          "REMGEN",
          "NPORT",
          "BadT",
          "SSD",
          "ASSD",
          "ASSDR",
          "ALFSSD",
          "XALFSSD",
          "Ascent",
          "NoUnlearning",
      ],
      help="Method for unlearning",
  )
  parser.add_argument(
      "--num_classes",
      type=int,
      default=10,
      choices=[2, 10, 100, 200],
      help="Number of Classes",
  )
  parser.add_argument(
      "--forget_set_size",
      type=int,
      default=500,
      help="Number of samples to be manipulated",
  )
  parser.add_argument(
      "--patch_size",
      type=int,
      default=3,
      help=(
          "Creates a patch of size patch_size x patch_size for poisoning at"
          " bottom right corner of image"
      ),
  )
  parser.add_argument(
      "--deletion_size",
      type=int,
      default=None,
      help="Number of samples to be deleted",
  )
  # Method Specific Params
  parser.add_argument(
      "--k",
      type=int,
      default=-1,
      help=(
          "All layers are freezed except the last-k layers, -1 means unfreeze"
          " all layers"
      ),
  )
  parser.add_argument(
      "--factor", type=float, default=0.1, help="Magnitude to decrease weights"
  )
  parser.add_argument(
      "--kd_T",
      type=float,
      default=4,
      help="Knowledge distilation temperature for SCRUB",
  )
  parser.add_argument(
      "--alpha",
      type=float,
      default=0.001,
      help=(
          "KL from og_model constant for SCRUB, higher incentivizes closeness"
          " to ogmodel"
      ),
  )
  parser.add_argument(
      "--msteps",
      type=int,
      default=400,
      help="Maximization steps on forget set for SCRUB",
  )
  parser.add_argument(
      "--SSDdampening",
      type=float,
      default=1.0,
      help="SSD: lambda aka dampening constant, lower leads to more forgetting",
  )
  parser.add_argument(
      "--SSDselectwt",
      type=float,
      default=10.0,
      help="SSD: alpha aka selection weight, lower leads to more forgetting",
  )
  parser.add_argument(
      "--rsteps",
      type=int,
      default=800,
      help="InfRe when to stop retain set gradient descent",
  )
  parser.add_argument(
      "--ascLRscale",
      type=float,
      default=1.0,
      help="AL/InfRe: scaling of lr to use for gradient ascent",
  )
  parser.add_argument(
      "--p_cut",
      type=float,
      default=0.0,
      help="cutoff fraction for adaptive SSD versions",
  )
  parser.add_argument(
      "--optim",
      type=str,
      default="SGD",
      choices=[
          "ADAM",
          "SGD",
      ],
      help="Optimizer",
  )
  # Optimizer Params
  parser.add_argument(
      "--batch_size",
      type=int,
      default=512,  # set in sh file
      help="input batch size for training (default: 512)",
  )
  parser.add_argument(
      "--pretrain_iters",
      type=int,
      default=7500,  # set in sh file
      help="number of epostepschs to train",
  )
  parser.add_argument(
      "--unlearn_iters",
      type=int,
      default=1000,
      help="number of steps to train",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="seed",
  )
  parser.add_argument(
      "--unlearn_lr",
      type=float,
      default=0.025,  # 0.025, help="learning rate (default: 0.025)"
  )
  parser.add_argument(
      "--pretrain_lr",
      type=float,
      default=0.025,  # 25, #0.025,
      help="learning rate (default: 0.025)",
  )
  parser.add_argument(
      "--wd",
      type=float,
      default=0.0005,
      help=(  # wrongly described as learning rate in original paper
          "weight decay (default: 0.01, 0.0005)"
      ),
  )
  # Guiding memorization args
  parser.add_argument(
      "--train_type", type=str, default="baseline", help="type of training"
  )
  parser.add_argument(
      "--tsvpath",
      type=str,
      default="results.tsv",
      help="path of tsv file (dont add .tsv)",
  )
  parser.add_argument(
      "--dirpath",
      type=str,
      default="./logs/",
      help="root directory of saved results",
  )
  parser.add_argument(
      "--project_name",
      type=str,
      default="prjname",
      help="wandb project name",
  )
  # Defaults
  parser.add_argument("--data_dir", type=str, default="data/")
  parser.add_argument("--save_dir", type=str, default="logs/")
  parser.add_argument("--exp_name", type=str, default="unlearn")
  parser.add_argument(
      "--device",
      type=str,
      default="cuda" if torch.cuda.is_available() else "mps",
  )
  args = parser.parse_args()
  return args
