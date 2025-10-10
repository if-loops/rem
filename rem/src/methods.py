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

"""Unlearning method functions and classes.

Adapted from "Corrective Unlearning" and
"Potion:Towards Poison Unlearning" GitHub repos.
https://github.com/drimpossible/corrective-unlearning-bench
https://github.com/if-loops/towards_poison_unlearning

To add a new unlearning or pretraining method,
create a class in this file.
Then add the class name in opts.py as a choice,
as well as in main.py (line 530 onwards).
"""

import copy
import datetime
import os
from os import path
import random
import re
import time
from typing import Any, Dict, List
import config
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch.utils import tensorboard
from torch.utils.data import sampler
import torchmetrics
import tqdm
from utils import alfssd_tuning
from utils import assd_tuning
from utils import distill_kl_loss
from utils import LinearLR
from utils import ssd_tuning
from utils import unlearn_func

makedirs = os.makedirs
exists = path.exists
randint = random.randint
SubsetRandomSampler = sampler.SubsetRandomSampler
SummaryWriter = tensorboard.SummaryWriter
DataLoader = data.DataLoader
GradScaler = torch.amp.GradScaler
autocast = torch.amp.autocast
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # replace with MPS if running on Mac
# Potion setup
ITERATIVE_SEARCH = config.iterative_search_potion  # iterative search of potion
STEP_MULT = config.step_mult_potion  # Potion parameter taken from original repo
MAX_TRY = config.max_try_potion  # Potion parameter taken from original repo
# Specify the file names for the importances
# to not keep them in memory (unused but can be used if OOM occurs)
file_name_1 = "original_importances.pkl"
file_name_2 = "sample_importances.pkl"


class ExampleTiedDropout(torch.nn.Module):
  """A dropout layer that ties neurons across a set of examples.

  This class is similar to batch tied dropout, but instead of tying neurons in a
  batch, it ties neurons in a set of examples. Based on:
  https://github.com/pratyushmaini/localizing-memorization/blob/b09ba96e412c03301ddd0b944b511495d863c60b/models/dropout.py#L37
  Depending on the set drop mode for the model (set in Experiments_run.py), ETD
  will behave differently as described in the REM paper (link in Readme.md)
  """

  def __init__(
      self,
      p_fixed=None,
      p_mem=None,
      num_batches=None,
      drop_mode="NO DROP MODE SET",
      max_id=60000,
  ):  # maxid 60000 for cifar
    super(ExampleTiedDropout, self).__init__()
    self.vit = False
    # self.seed = 101010
    self.max_id = (
        max_id  # maxid hardcoded as cifar10 and cifar100 both have 60k samples
    )
    self.p_mem = p_mem
    self.p_fixed = p_fixed
    self.p_fold = p_fixed
    self.p_mem_org = p_mem
    self.p_fixed_org = p_fixed
    self.counter_rand = 0
    self.drop_mode = drop_mode
    self.train_drop_gen = False
    self.idx_dict = {}
    # self.config = config cannot be pickled
    self.rand_dropout_on = config.rand_dropout_on
    self.rand_dropout_p = config.rand_dropout_p
    # really ugly
    try:  # avoids overwriting
      self.mask_tensor is not None
    except AttributeError:
      self.mask_tensor = None
    self.mask_ready = torch.zeros(self.max_id).to(device)
    self.dropped = False
    self.drop_targets = None
    self.count_mask = None
    self.classidx = None
    self.epoch_i = 0
    self.force_mask_id = 42
    self.max_pairs = 4
    self.mask_tensor_dict = {}
    self.mask_ready_dict = {}
    self.mask_tensor = []
    self.count_mask_tensor = []
    for i in range(self.max_pairs):
      self.mask_ready_dict[i] = torch.zeros(self.max_id).to(device)
      self.mask_tensor_dict[i] = None
      self.mask_tensor.append(None)
      self.count_mask_tensor.append(None)
    self.etd_device = "cuda"
    self.etd_mult_device = "cuda"
    self.shuffle_mask = config.shuffle_mask
    self.ft_end = config.ft_end  # 0 for off
    self.force_all = config.force_all
    print("DEVICE: ", self.etd_device)
    print("Using ETD with p_mem, p_fixed:", p_mem, p_fixed)

  def shuffle_deterministic(self, tensor, seed=0):
    """Shuffles the values of a PyTorch tensor along the second dimension deterministically.

    Args:
        tensor: The input tensor.
        seed: The seed for the random number generator. Defaults to 42.

    Returns:
        A new tensor with the second dimension shuffled deterministically.
    """
    # does not change results
    # Get the number of elements in the second dimension
    num_elements = tensor.size(1)
    # Generate a deterministic permutation of indices
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(num_elements, generator=generator)
    # Index the tensor with the shuffled indices
    shuffled_tensor = tensor[:, indices]
    return shuffled_tensor

  def no_id_overflow(self, mask_id):
    # safety to prevent overflow and reuse IDs
    # (otherwise eternal OOM or slow due to read write to disk)
    if mask_id > self.max_id:
      while mask_id > self.max_id:
        mask_id -= self.max_id
    return mask_id  # starts to count from start again

  def forward(self, x, idx, epoch=0, classidx=None, pair=0):
    if pair not in config.pairs:
      return x
    x = x.to(self.etd_mult_device)
    torch.cuda.empty_cache()
    self.mask_ready = self.mask_ready_dict[pair]
    # self.mask_tensor = self.mask_tensor_dict[pair]
    self.classidx = classidx  # used for v5
    self.epoch_i = epoch
    if self.p_fixed == 1:
      return x
    if (
        self.drop_mode == "train"
        or self.drop_mode == "random"
        or self.drop_mode == "train_one_ul_mask"
        or self.drop_mode == "reverse_drop"
    ):
      idx = idx.to(self.etd_device)
      idx_mod = copy.deepcopy(idx)
      count_in = 0
      count_all = 0
      if (
          self.drop_mode == "train_one_ul_mask"
          or self.drop_mode == "reverse_drop"
      ):
        # to give the same mask to all identified samples
        for n_i, i in enumerate(idx):
          # n_i = self.no_id_overflow(n_i)
          if config.rem_mod_force_df_singlemask:
            count_all += 1
            if i.cpu() in self.drop_targets:
              count_in += 1
              idx_mod[n_i] = self.force_mask_id  # set all manips to same mask
            else:
              if config.rem_mod_flip_channel_coin > 0:
                if torch.rand(1) > (1 - config.rem_mod_flip_channel_coin):
                  idx_mod[n_i] = torch.randperm(config.rem_mod_channels)[0]
              elif config.rem_mod_rand_nomanip_mask:
                rand_idx = randint(0, 50000)
                idx_mod[n_i] = rand_idx
              else:
                pass
          else:
            idx_mod[n_i] = (
                self.force_mask_id
            )  # all use same single mask for channeling
      # idx = idx.to(self.etd_device)
      idx_mod = idx_mod.to(self.etd_device)
      # create a mask based on the index (idx)
      mask = torch.zeros_like(x).to(self.etd_device)
      shape = x.shape[1]
      # print("X Shape--: ", x.shape)
      if torch.all(self.mask_ready[idx_mod] == 1):
        if self.drop_mode == "random":
          rand_idx = randint(0, 50000)
          mask = self.mask_tensor[pair][rand_idx]
        else:  # all others
          mask = self.mask_tensor[pair][idx_mod]
      else:  # elif torch.all(self.mask_ready[idx] == 0): #epoch == 0:
        # print(pair)
        self.mask_ready[idx_mod] = 1
        self.mask_ready_dict[pair] = self.mask_ready
        # print("epoch", epoch, "mode", self.drop_mode)
        # keep all neurons with index less than self.p_fixed*shape
        mask[:, : int(self.p_fixed * shape)] = 1
        # Fraction of elements to keep
        p_mem = self.p_mem
        # Generate a random mask for each row in the input tensor
        shape_of_mask = shape - int(self.p_fixed * shape)
        for i in range(x.shape[0]):
          torch.manual_seed(idx_mod[i].item())
          curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
          # repeat curr_mask along dimension 2 and 3 to have the same shape as X
          if not self.vit:
            curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1)
          else:
            curr_mask = curr_mask.unsqueeze(-1)  # .unsqueeze(-1)
          mask[i][int(self.p_fixed * shape) :] = curr_mask
        if self.mask_tensor[pair] is None:
          if not self.vit:
            self.mask_tensor[pair] = torch.zeros(
                self.max_id,
                x.shape[1],
                x.shape[2],
                x.shape[3],
                dtype=torch.uint8,
            ).to(self.etd_device)
          else:
            self.mask_tensor[pair] = torch.zeros(
                self.max_id, x.shape[1], x.shape[2], dtype=torch.uint8
            ).to(self.etd_device)
        mask = mask.to(torch.uint8)
        # self.mask_tensor = self.mask_tensor.cpu()
        self.mask_tensor[pair][idx_mod] = mask.to(
            self.etd_device
        )  # .to(torch.uint8)
      if config.train_flip_noise:
        # randomly flip x% of the zeros to 1
        p_flip = 0.01
        mask = torch.where(
            (mask == 0) & (torch.rand_like(mask.float()) < p_flip), 1.0, mask
        ).to(torch.uint8)
      if self.shuffle_mask:
        mask = self.shuffle_deterministic(
            mask
        )  # to ensure gen is spread across the network and not in one spot
      if config.rand_dropout_on:
        pass
      elif config.p_scale:
        x.mul_(mask.to(x.device))
        if self.p_mem != 0:
          x = x * (1 - self.p_fixed) * (1 - self.p_mem)
        else:
          print(
              "==> Running with p_mem=0. This is not intended. Falling back to"
              " default ETD behaviour."
          )
      else:
        x.mul_(mask.to(x.device))  # save mem to prevent OOM
      if config.upscale_mem_during_train and (self.p_mem != 0):
        assert not self.shuffle_mask  # Does not work with shuffle mask
        x[:, int(self.p_fixed * shape) :] = x[
            :, int(self.p_fixed * shape) :
        ] / (self.p_mem * 2)
      if self.drop_mode == "reverse_drop":  # removes the gen part
        x[:, : int(self.p_fixed * shape)] = 0
        x[:, int(self.p_fixed * shape) :] = x[:, int(self.p_fixed * shape) :]
      self.counter_rand += 1
      del mask
    elif self.drop_mode == "test":
      # At test time we will renormalize outputs from the non-fixed neurons
      # based on the number of neuron sets
      # we will keep the fixed neurons unmodified
      shape = x.shape[1]
      if (config.p_scale) and (self.p_mem != 0) and (self.p_mem != 1.0):
        tail_comp = config.tail_comp
        test_mem = self.p_mem * (1 - self.p_fixed) / tail_comp
      else:
        test_mem = self.p_mem
      # Finetuning with all neurons to avoid rescaling
      if self.ft_end != 0:
        if self.epoch_i:
          if self.epoch_i > self.ft_end:
            if (self.p_mem != 0) and (self.p_fixed != 0):
              return x
          else:
            pass
        else:
          if (self.p_mem != 0) and (self.p_fixed != 0):
            return x
      if self.force_all:
        return x
      if self.shuffle_mask:
        dummy_full = torch.ones_like(x, dtype=torch.half)
        dummy_full[:, int(self.p_fixed * shape) :] = (
            dummy_full[:, int(self.p_fixed * shape) :] * test_mem
        )
        dummy_full = self.shuffle_deterministic(dummy_full)
        x.mul_(dummy_full.to(x.device))
      else:
        x[:, : int(self.p_fixed * shape)] = x[:, : int(self.p_fixed * shape)]
        x[:, int(self.p_fixed * shape) :] = (
            x[:, int(self.p_fixed * shape) :] * test_mem
        )  # just changes the magnitude! does not drop anything
    elif self.drop_mode == "drop":
      tail_comp = 1
      if (self.p_mem != 0) and (self.p_mem != 1.0):
        if config.drop_no_scaling:
          drop_mem = 0
        elif config.p_scale:
          drop_mem = self.p_mem * (1 - self.p_fixed)
          tail_comp = config.tail_comp
        else:
          drop_mem = self.p_mem
      elif self.p_mem == 1.0:
        drop_mem = 0
      else:
        drop_mem = self.p_mem
      shape = x.shape[1]
      if self.shuffle_mask:
        dummy_full = torch.ones_like(x, dtype=torch.half)
        dummy_full[:, int(self.p_fixed * shape) :] = (
            dummy_full[:, int(self.p_fixed * shape) :] * 0
        )  # set to drop out
        dummy_full[:, : int(self.p_fixed * shape)] = (
            dummy_full[:, : int(self.p_fixed * shape)]
            * (self.p_fixed + drop_mem)
            / self.p_fixed
        )
        dummy_full = self.shuffle_deterministic(dummy_full)
        x.mul_(dummy_full.to(x.device))
      else:
        shape = x.shape[1]
        x[:, int(self.p_fixed * shape) :] = 0
        x[:, : int(self.p_fixed * shape)] = (
            x[:, : int(self.p_fixed * shape)]
            * (self.p_fixed + drop_mem)
            / self.p_fixed
            * tail_comp
        )
    elif self.drop_mode == "drop_gen":
      mask = self.mask_tensor[pair][idx]
      x.mul_(mask.to(x.device))
      # New
      shape = x.shape[1]
      if self.shuffle_mask:
        dummy_full = torch.ones_like(x, dtype=torch.half)
        dummy_full[:, int(self.p_fixed * shape) :] = dummy_full[
            :, int(self.p_fixed * shape) :
        ]  # set to drop out
        dummy_full[:, : int(self.p_fixed * shape)] = (
            dummy_full[:, : int(self.p_fixed * shape)] * 0
        )  # *(self.p_fixed + self.p_mem)/self.p_fixed
        dummy_full = self.shuffle_deterministic(dummy_full)
        x.mul_(dummy_full.to(x.device))
      else:
        shape = x.shape[1]
        x[:, int(self.p_fixed * shape) :] = x[:, int(self.p_fixed * shape) :]
        x[:, : int(self.p_fixed * shape)] = 0
    elif self.drop_mode == "idealtrain":
      # Gives full knowledge of corruptions to ETD
      # "Cheats" this way to give an upper bound on the performance
      idx = idx.cpu()
      # create a mask based on the index (idx)
      mask = torch.zeros_like(x).to(self.etd_device)
      shape = x.shape[1]
      # print("X Shape--: ", x.shape)
      # THE IDEAL PART TO SHOW UPPER LIMIT
      idx_mod = copy.deepcopy(idx)
      try:
        for n_i, i in enumerate(idx):
          if i in self.drop_targets:
            idx_mod[n_i] = 0  # set all manips to mask 0
          else:
            # idx_mod[n_i] = 0
            pass
      except TypeError as e:
        print(f"forget idx not provided: {e}")
      idx = idx.to(self.etd_device)
      idx_mod = idx_mod.to(self.etd_device)
      if torch.all(self.mask_ready[idx] == 1):  # epoch > 0:
        # get mask from self.mask_tensor
        # mask = self.mask_tensor[pair, idx.cpu()].cpu()
        mask = self.mask_tensor[pair][idx]
      elif torch.all(self.mask_ready[idx] == 0):  # epoch == 0:
        self.mask_ready[idx] = 1
        self.mask_ready_dict[pair] = self.mask_ready
        # print("epoch", epoch, "mode", self.drop_mode)
        # keep all neurons with index less than self.p_fixed*shape
        mask[:, : int(self.p_fixed * shape)] = 1
        # Fraction of elements to keep
        p_mem = self.p_mem
        # Generate a random mask for each row in the input tensor
        shape_of_mask = shape - int(self.p_fixed * shape)
        for i in range(x.shape[0]):
          torch.manual_seed(idx_mod[i].item())
          curr_mask = torch.bernoulli(torch.full((1, shape_of_mask), p_mem))
          # repeat curr_mask along dimension 2 and 3 to have the same shape as X
          curr_mask = curr_mask.unsqueeze(-1).unsqueeze(-1)
          mask[i][int(self.p_fixed * shape) :] = curr_mask
        if self.mask_tensor[pair] is None:
          self.mask_tensor[pair] = torch.zeros(
              self.max_id, x.shape[1], x.shape[2], x.shape[3], dtype=torch.uint8
          ).to(self.etd_device)
        # assign mask at positions given by idx
        # print("tensor comp:", self.mask_tensor.shape, mask.shape)
        # mask = mask.to(torch.uint8)
        self.mask_tensor[pair][idx.to(self.mask_tensor[pair].device)] = mask.to(
            torch.uint8
        ).to(self.mask_tensor[pair].device)
        # self.mask_tensor[idx] = mask.cpu().float() # too big for GPU VRAM
      if self.shuffle_mask:
        mask = self.shuffle_deterministic(mask)
      # Apply the mask to the input tensor
      x.mul_(mask.to(x.device))
    x = x.to(device)
    return x


class Naive:
  """Base class for unlearning methods.

  This class provides a framework for training and evaluating models, including
  functionalities for forward passes, optimizer and scheduler setup, and result
  saving. It also includes methods for reinitializing model layers. Specific
  unlearning methods can inherit from this class and override the `unlearn` and
  `forward_pass` methods.
  """

  def __init__(
      self, opt, model, prenet=None, train_type="NONE GIVEN", log_name="None"
  ):
    self.lossdrop = False
    self.preul_train_type = model.ttype_store
    print("self.preul_train_type: ", self.preul_train_type)
    if "LLF" in train_type:
      train_type_split = train_type.split("_")
      relevance_part = [part for part in train_type_split if "LLF" in part][0]
      match_e = re.search(r"LLFe(\d+)", relevance_part)
      match_l = re.search(r"l(\d+)", relevance_part)
      if match_e is None:
        raise ValueError(
            f"Could not find 'LLFe(\\d+)' in relevance_part: {relevance_part}"
        )
      self.unlearn_e = int(
          match_e.group(1)
      )  # Number of epochs between reinitialization
      if match_l is None:
        raise ValueError(
            f"Could not find 'l(\\d+)' in relevance_part: {relevance_part}"
        )
      self.unlearn_n = int(match_l.group(1))  # n last layers to reinitialize
    if "ETD" in train_type:  # Added ETD case
      pass
    if "CLIP" in train_type:
      self.clip = True
    else:
      self.clip = False
    if train_type == "UL":
      print("Running UL on pretrained model")
    if train_type == "NONE GIVEN":
      assert False
    current_time = datetime.datetime.now().strftime(
        "%d%m%Y-%H%M%S"
    )  # Format: DDMMYYYY-HHMMSS
    note = "debug_"
    log_dir = f"runs/{note}{current_time}_{log_name}"
    self.writer = SummaryWriter(log_dir)
    self.opt = opt
    self.train_type = train_type
    self.curr_step, self.best_top1 = 0, 0
    self.best_model = None
    self.set_model(model, prenet)
    self.save_files: Dict[str, List[Any]] = {
        "train_top1": [],
        "val_top1": [],
        "train_time_taken": [0.0],
    }
    if self.opt.optim == "SGD":
      self.optimizer = torch.optim.SGD(
          self.model.parameters(),
          lr=self.opt.max_lr,
          momentum=0.9,
          weight_decay=self.opt.wd,
      )
      self.scheduler = LinearLR(
          self.optimizer,
          t=self.opt.train_iters * 1.25,
          warmup_epochs=self.opt.train_iters // 100,
      )  # Spend 1% time in warmup, and stop 66% of the way through training
    elif self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    self.top1 = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    self.scaler = GradScaler()

  def set_model(self, model, prenet=None):
    _ = prenet
    self.prenet = None
    self.model = model
    self.model.to(device)
    # self.model.to(device)

  def forward_pass(self, images, target, idx, epoch, infgt, flipsign=False):
    _ = infgt
    _ = flipsign
    output = self.model(images, idx, epoch, target)
    loss = F.cross_entropy(output, target)
    self.top1(output, target)
    return loss

  def reinitialize_last_layers(self):
    """Reinitializes the weights and biases of the last n layers of the model.

    The number of layers to reinitialize is determined by `self.unlearn_n`.
    Kaiming Normal initialization is used for weights, and biases to zero.
    """
    # Get the number of layers in the model
    num_layers = len(list(self.model.children()))
    # print("Model layers:", list(self.model.children()))
    # Get the starting index for reinitialization
    reinit_start = (
        num_layers - self.unlearn_n
    )  # max(round(num_layers * self.unlearn_n), 1)
    # Loop through the layers and reinitialize weights/biases if needed
    for i, module in enumerate(self.model.children()):
      if i >= reinit_start:
        # Apply weight/bias initialization (e.g., using Kaiming initialization)
        if hasattr(module, "weight"):
          torch.nn.init.kaiming_normal_(module.weight)
        if hasattr(module, "bias") and module.bias is not None:
          torch.nn.init.zeros_(module.bias)
    print(
        f"Reinitialized the last {self.unlearn_n} layers of {num_layers} at"
        f" epoch {self.curr_epoch}"
    )

  def reinitialize_gen_in_last_n_layers(
      self,
  ):
    """Reinitializes the 'generalization' part of the last n layers.

    ETD is counted as layer 8, so always add an extra layer for resetting to
    skip it. The number of layers to reinitialize is determined by
    `self.unlearn_n`. Kaiming Normal initialization is used for the weights in
    the generalization part, and biases are set to zero.
    """
    # Get the number of layers in the model
    num_layers = len(list(self.model.children()))
    # print("Model layers:", list(self.model.children()))
    # Get the starting index for reinitialization
    reinit_start = (
        num_layers - self.unlearn_n
    )  # max(round(num_layers * self.unlearn_n), 1)
    # Loop through the layers and reinitialize weights/biases if needed
    for i, module in enumerate(self.model.children()):
      if i >= reinit_start:
        # Apply weight/bias initialization (e.g., using Kaiming initialization)
        if hasattr(module, "weight"):
          num_weights = module.weight.numel()
          gen_stop = int(self.model.dropout.p_fixed * num_weights)
          torch.nn.init.kaiming_normal_(module.weight[:gen_stop])
        if hasattr(module, "bias") and module.bias is not None:
          num_bias = module.weight.numel()
          gen_stop = int(self.model.dropout.p_fixed * num_bias)
          torch.nn.init.zeros_(module.bias[:gen_stop])
    print(
        f"Reinitialized the last {self.unlearn_n} layers of {num_layers} at"
        f" epoch {self.curr_epoch}"
    )

  def train_one_epoch(
      self,
      loader,
      ascent=False,
      forced_target=None,
      delete_idx=None,
      no_scheduler_step=False,
      flipsign=False,
  ):
    """Trains the model for one epoch.

    Args:
        loader (DataLoader): The data loader for the training data.
        ascent (bool, optional): If True, performs gradient ascent. Defaults to
          False.
        forced_target (int, optional): If not None, forces the target of
          `delete_idx` samples to this value. Defaults to None.
        delete_idx (list, optional): A list of indices to apply `forced_target`
          to. Defaults to None.
        no_scheduler_step (bool, optional): If True, prevents the scheduler from
          stepping. Defaults to False.
        flipsign (bool, optional): Flag to modify loss calculation, specific to
          certain unlearning methods. Defaults to False.

    Returns:
        tuple: A tuple containing the average epoch loss and the top-1 accuracy.
    """
    self.model.train()
    self.top1.reset()
    # Check if reinitialization is needed for this epoch
    if "LLF" in self.train_type:
      if self.curr_epoch % self.unlearn_e == 0:
        # safety to not reset just before end
        if self.curr_step < 0.95 * self.opt.train_iters:
          self.reinitialize_gen_in_last_n_layers()
          # self.reinitialize_last_layers() # ORIGINAL LLF
    epoch_loss = 0.0  # Initialize epoch loss accumulator
    self.gradient_mask = None
    for images, target, infgt, idx in tqdm.tqdm(loader):
      if forced_target:
        if delete_idx:
          for i, idxi in enumerate(idx):
            if idxi in delete_idx:
              target[i] = forced_target
        else:
          pass
          # target = target*0+forced_target # forces the whole dataset to label
      images, target, infgt, idx = (
          images.to(device),
          target.to(device),
          infgt.to(device),
          idx.to(device),
      )
      with autocast(device_type=device.type):
        self.optimizer.zero_grad()
        losscomb = False
        batchnorm_trick = False
        if losscomb:
          # once with the intended drop mode
          og_mode = self.model.dropout.drop_mode
          loss_train = self.forward_pass(
              images, target, idx, self.curr_epoch, infgt
          )
          # self.scaler.scale(loss_train).backward()
          # once with all neurons firing
          self.model.dropout.drop_mode = "test"
          loss_test = self.forward_pass(
              images, target, idx, self.curr_epoch, infgt
          )
          self.scaler.scale(loss_train / 2 + loss_test / 2).backward()
          self.model.dropout.drop_mode = og_mode  # set back
          loss = loss_test + loss_train
          epoch_loss += loss.item()  # Accumulate loss for each batch
        elif batchnorm_trick:  # also set test
          self.model.train()  # collect batch norm statistics
          og_mode = self.model.dropout.drop_mode
          self.model.dropout.drop_mode = "test"
          self.model.eval()
          self.model.dropout.drop_mode = "train"
          loss = self.forward_pass(images, target, idx, self.curr_epoch, infgt)
          self.model.train()
          epoch_loss += loss.item()  # Accumulate loss for each batch
          self.scaler.scale(loss).backward()
          # reset everything
          self.model.dropout.drop_mode = og_mode
        else:
          loss = self.forward_pass(
              images, target, idx, self.curr_epoch, infgt, flipsign
          )
          if ascent:
            loss = -loss  # gradient ascent
          epoch_loss += loss.item()  # Accumulate loss for each batch
          self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        # print("SCHEDULER LR before step: ",self.scheduler.get_lr())
        try:  # prevents code breaking if more steps than scheduler
          if no_scheduler_step:
            self.curr_step -= 1  # cancel out the step update
          elif self.opt.optim == "ADAM":
            pass
          else:
            self.scheduler.step()
        except RuntimeError:
          # This can happen if the scheduler has no more steps.
          pass
        self.curr_step += 1
        if self.curr_step > self.opt.train_iters:
          break
    top1 = float(self.top1.compute())  # .item() at end works too
    # but linter complains. Thus use of float()
    self.top1.reset()
    self.save_files["train_top1"].append(top1)
    print(f"Step: {self.curr_step} Train Top1: {top1:.3f}")
    average_epoch_loss = epoch_loss / len(loader)
    return average_epoch_loss, top1

  def eval(
      self,
      loader,
      save_model=True,
      save_preds=False,
      train_fix=False,
      class_0=False,
      ic=False,
      icmanip=False,
      only_infgti=False,
      no_manip=False,
  ):
    """Evaluates the model on a given data loader.

    Args:
        loader (DataLoader): The data loader to evaluate on.
        save_model (bool, optional): Whether to save the model if it achieves
          the best top-1 accuracy. Defaults to True.
        save_preds (bool, optional): Whether to return predictions and targets.
          Defaults to False.
        train_fix (bool, optional): If True, expects loader to yield (images,
          target, infgt, idx) and applies specific logic for
          manipulated/non-manipulated samples. Defaults to False.
        class_0 (bool, optional): If True, forces all targets to 0. Defaults to
          False.
        ic (bool, optional): If True, applies class-swap logic for evaluation.
          Defaults to False.
        icmanip (bool, optional): Similar to `ic`, used within `train_fix`.
          Defaults to False.
        only_infgti (bool, optional): Used with `train_fix`. If True, only
          evaluates on samples where `infgt` is 1. Defaults to False.
        no_manip (bool, optional): Used with `train_fix`. If True, only
          evaluates on samples where `infgt` is 0. Defaults to False.

    Returns:
        float or tuple: If `save_preds` is False, returns the top-1 accuracy.
          If `save_preds` is True, returns a tuple of (predictions, targets).
    """
    loader_call = loader  # copy.deepcopy(loader)
    self.model.eval()
    self.top1.reset()
    preds, targets = [], []
    with torch.no_grad():
      if train_fix:  # additional _ to unpack
        for images, target, infgti, idxi in tqdm.tqdm(loader_call):
          with autocast(device_type=device.type):
            # images, target = images.to(device), target.to(device)
            images, target = images.to(device), target.to(device)
            if only_infgti:
              mask = infgti == 1  # Create boolean tensor for comparison
              images = images[mask]
              target = target[mask]
              idxi = idxi[mask]  # only keep manip
            if no_manip:
              mask = infgti == 0  # Create boolean tensor for comparison
              images = images[mask]
              target = target[mask]
              idxi = idxi[mask]  # only non keep manip
            if ic or icmanip:  # Simplified check for ic
              if ic:
                unique_values = torch.unique(target)
                if len(unique_values) == 2:
                  val1, val2 = unique_values[0].item(), unique_values[1].item()
                  target[target == val1] = 999
                  target[target == val2] = val1
                  target[target == 999] = val2
                else:
                  pass
                  # print("#### ONLY ONE CLASS FROM SWAP IN BATCH")
            if class_0:
              target *= 0  # REMOVE
            output = (
                self.model(images, idxi)
                if self.prenet is None
                else self.model(self.prenet(images))
            )
          try:
            self.top1(output, target)
          except ValueError:
            print(
                "Loader empty (e.g. in case of EU with no adversarial loader)"
            )
          if save_preds:
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
      else:
        for images, target in tqdm.tqdm(loader_call):
          with autocast(device_type=device.type):
            # images, target = images.to(device), target.to(device)
            images, target = images.to(device), target.to(device)
            if class_0:
              target *= 0  # REMOVE
            if self.prenet is None:
              output = self.model(
                  images, torch.zeros_like(target, device=device)
              )  # for test time (idx placeholder)
            else:
              output = self.model(self.prenet(images))
          self.top1(output, target)
          if save_preds:
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
    top1 = float(self.top1.compute())  # .item()
    top1_return = top1
    self.top1.reset()
    if not save_preds:
      print(f"Step: {self.curr_step} Val Top1: {top1*100:.2f}")
    if save_model:
      # Ensure self.save_files["val_top1"] is a list before appending.
      if isinstance(self.save_files["val_top1"], float):
        self.save_files["val_top1"] = [self.save_files["val_top1"]]
      elif not isinstance(self.save_files["val_top1"], list):
        self.save_files["val_top1"] = []
      self.save_files["val_top1"].append(top1)
      if top1 > self.best_top1:
        self.best_top1 = top1
        self.best_model = copy.deepcopy(self.model).cpu()
    self.model.train()
    if save_preds:
      preds = np.concatenate(preds, axis=0)
      targets = np.concatenate(targets, axis=0)
      return preds, targets
    return top1_return

  def unlearn(
      self,
      train_loader,
      test_loader,
      eval_loaders=None,
      adversarial_train_loader=None,
      forget_idx=None,
      frac_dl=None,  # added for completness (linter)
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    """Orchestrates the unlearning process.

    This method handles the training loop for unlearning, including setting
    dropout modes based on the unlearning method and configuration, performing
    training epochs, and evaluating the model periodically. It also logs various
    metrics to TensorBoard.

    Args:
        train_loader (DataLoader): DataLoader for the training set (retain set).
        test_loader (DataLoader): DataLoader for the test set.
        eval_loaders (dict, optional): Dictionary of additional DataLoaders for
          evaluation. Defaults to None.
        adversarial_train_loader (DataLoader, optional): DataLoader for
          adversarial examples in the training set. Defaults to None.
        forget_idx (list, optional): Indices of samples to be forgotten.
          Defaults to None.
        frac_dl (float, optional): Fraction of data to dampen. Defaults to None.
        min_acc_val (float, optional): Minimum accuracy value. Defaults to None.
        pretrain_loader (DataLoader, optional): DataLoader for pretraining data.
          Defaults to None.
        delete_idx (list, optional): Indices of samples to be deleted. Defaults
          to None.
        eval_train (DataLoader, optional): DataLoader for training evaluation.
          Defaults to None.
    """
    self.curr_epoch = 0
    _ = eval_loaders
    _ = forget_idx
    _ = frac_dl
    _ = min_acc_val
    _ = pretrain_loader
    _ = delete_idx
    _ = eval_train
    if "ETD" in self.preul_train_type:
      train_type_split = self.preul_train_type.split("_")
      try:
        relevance_part = [part for part in train_type_split if "ETD" in part][0]
      except ValueError:
        relevance_part = [part for part in train_type_split if "ETD" in part]
      match_drop = re.search(r"drop(\d+)", relevance_part)
      self.match_drop = str(match_drop[1])  # [1] replaces .group(1)
      # to please the linter. can be used interchangebly
    elif "FREEZE" in self.preul_train_type:
      match_drop = re.search(r"drop(\d+)", self.preul_train_type)
      self.match_drop = str(match_drop[1])
    else:
      self.match_drop = "NO ETD GIVEN"  # will raise error if ETD is still used
    # pass via model for UL methods
    self.model.match_drop = self.match_drop
    print("--------")
    print("--------")
    print(
        f"Current: {self.train_type}; Pretrained with {self.preul_train_type};"
        f" ETD mode: {self.match_drop}"
    )
    print("--------")
    print("--------")
    while self.curr_step < self.opt.train_iters:
      time_start = time.process_time()
      if self.train_type == "UL":
        self.lossdrop = False  # only for pretrain
        if self.match_drop == "0":  # pass all neurons through
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "1":  # drop all mem neurons
          self.model.dropout.drop_mode = "drop"
        elif self.match_drop == "2":  # extra option to be used
          self.model.dropout.drop_mode = "drop"
        elif self.match_drop == "3":  # extra option to be used
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "4":  # extra option to be used
          self.model.dropout.drop_mode = "targetdrop"
        elif self.match_drop == "5":  # extra option to be used
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "9":  # extra option to be used
          self.model.dropout.drop_mode = "train"
        elif self.match_drop == "8":  # extra option to be used
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = "test"
      else:
        if self.match_drop == "3":  # extra option to be used
          self.model.dropout.drop_mode = "idealtrain"
        elif self.match_drop == "4":  # extra option to be used
          self.model.dropout.drop_mode = "idealtrain"
        elif self.match_drop == "0":  # extra option to be used
          self.model.dropout.drop_mode = "train"
        elif self.match_drop == "1":  # extra option to be used
          self.model.dropout.drop_mode = "train"
        elif self.match_drop == "2":  # extra option to be used
          self.model.dropout.drop_mode = "idealtrain"
        elif self.match_drop == "5":  # extra option to be used
          self.model.dropout.drop_mode = "classetd"
        elif self.match_drop == "8":  # extra option to be used
          self.model.dropout.drop_mode = "onlyoutlier"
        elif self.match_drop == "9":  # extra option to be used
          self.model.dropout.drop_mode = "train"
        else:
          self.model.dropout.drop_mode = (  # only train for the backwards pass
              "PRETRAIN METHOD WITHOUT ETD"
          )
      if (self.train_type == "UL") and (config.etd_ul):
        prev_mode = self.model.dropout.drop_mode
        self.model.dropout.drop_mode = "train_one_ul_mask"
        print("UL SPECIAL drop_mode (train): ", self.model.dropout.drop_mode)
        avg_loss, acc_top = self.train_one_epoch(loader=train_loader)
        self.model.dropout.drop_mode = (
            prev_mode  # setting back for eval/inference
        )
        print(
            "UL SPECIAL drop_mode set back to inference mode (test): ",
            self.model.dropout.drop_mode,
        )
      else:
        print("drop_mode (train): ", self.model.dropout.drop_mode)
        avg_loss, acc_top = self.train_one_epoch(loader=train_loader)
      if self.train_type != "UL":
        # only needed for ETD (redundant for rest as already set)
        if self.match_drop == "0":  # pass all neurons through
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "1":  # drop all mem neurons
          self.model.dropout.drop_mode = "drop"
        elif self.match_drop == "2":  # extra option to be used
          self.model.dropout.drop_mode = "drop"
        elif self.match_drop == "3":  # extra option to be used
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "4":  # extra option to be used
          self.model.dropout.drop_mode = "targetdrop"
        elif self.match_drop == "5":  # extra option to be used
          self.model.dropout.drop_mode = "test"
        elif self.match_drop == "8":  # extra option to be used
          self.model.dropout.drop_mode = "drop"
        elif self.match_drop == "9":  # extra option to be used
          self.model.dropout.drop_mode = "train"
        else:
          self.model.dropout.drop_mode = "test"
      print("drop_mode (test):", self.model.dropout.drop_mode)
      self.model.eval()
      if (self.curr_epoch % 100 == 0) and (
          self.curr_epoch > 0
      ):  # logging frequency can be changed here
        if "ETD" in self.preul_train_type and self.train_type != "UL":
          preserve_mode = copy.deepcopy(self.model.dropout.drop_mode)
          # drop_mode_opts = ["train", "test", "drop", "targetdrop", "random"]
          drop_mode_opts = ["drop"]  # speed up by logging less
          for dropping_mode in drop_mode_opts:
            og_mem = self.model.dropout.p_mem
            if (dropping_mode == "test") and (
                self.train_type != "UL"
            ):  # add different rescaling
              if self.model.dropout.p_mem == 1.0:
                mems = [0.0, self.model.dropout.p_mem]
              else:
                mems = [0.0, self.model.dropout.p_mem, 0.5, 1.0]
            else:
              mems = [self.model.dropout.p_mem]
            for mem_i in mems:
              self.model.dropout.p_mem = mem_i
              self.model.dropout.drop_mode = dropping_mode
              if not adversarial_train_loader and self.train_type != "UL":
                acc_adv_train = self.eval(
                    train_loader,
                    save_model=False,
                    train_fix=True,
                    class_0=False,
                    ic=True,
                    only_infgti=True,
                )
                # meaningless for randomlabel
                self.writer.add_scalar(
                    f"{dropping_mode}/manipulation_healed{mem_i}",
                    acc_adv_train,
                    self.curr_epoch,
                )
                acc_adv_train_red = self.eval(
                    train_loader,
                    save_model=False,
                    train_fix=True,
                    class_0=False,
                    ic=False,
                    icmanip=True,
                    only_infgti=True,
                )
                self.writer.add_scalar(
                    f"{dropping_mode}/manipulation_attack_success{mem_i}",
                    acc_adv_train_red,
                    self.curr_epoch,
                )
              elif adversarial_train_loader is not None:
                acc_adv_train = self.eval(
                    adversarial_train_loader,
                    save_model=False,
                    train_fix=False,
                    class_0=False,
                    only_infgti=True,
                )
                self.writer.add_scalar(
                    f"{dropping_mode}/poison_healed_rsc{mem_i}",
                    acc_adv_train,
                    self.curr_epoch,
                )
                acc_adv_train_red = self.eval(
                    adversarial_train_loader,
                    save_model=False,
                    train_fix=False,
                    class_0=True,
                    only_infgti=True,
                )
                self.writer.add_scalar(
                    f"{dropping_mode}/poison_attack_success{mem_i}",
                    acc_adv_train_red,
                    self.curr_epoch,
                )
              acc_test = self.eval(
                  test_loader, save_model=False, save_preds=False
              )  # save false is important or it will take
              # another drop mode than inteded for final results!
              self.writer.add_scalar(
                  f"{dropping_mode}/test_acc{mem_i}", acc_test, self.curr_epoch
              )
            self.model.dropout.p_mem = og_mem
          self.model.dropout.drop_mode = (
              preserve_mode  # Reset it to the expected mode
          )
        else:
          if not adversarial_train_loader and self.train_type != "UL":
            acc_adv_train = self.eval(
                train_loader,
                save_model=False,
                train_fix=True,
                class_0=False,
                ic=True,
                only_infgti=True,
            )
            self.writer.add_scalar(
                "Acc/manipulation_healed", acc_adv_train, self.curr_epoch
            )
            acc_adv_train_red = self.eval(
                train_loader,
                save_model=False,
                train_fix=True,
                class_0=False,
                ic=False,
                icmanip=True,
                only_infgti=True,
            )
            self.writer.add_scalar(
                "Acc/manipulation_attack_success",
                acc_adv_train_red,
                self.curr_epoch,
            )
          elif adversarial_train_loader is not None:
            acc_adv_train = self.eval(
                adversarial_train_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                only_infgti=True,
            )
            self.writer.add_scalar(
                "Acc/poison_healed", acc_adv_train, self.curr_epoch
            )
            acc_adv_train_red = self.eval(
                adversarial_train_loader,
                save_model=False,
                train_fix=False,
                class_0=True,
                only_infgti=True,
            )
            self.writer.add_scalar(
                "Acc/poison_attack_success", acc_adv_train_red, self.curr_epoch
            )
      if (self.curr_epoch % 5 == 0) and (self.curr_epoch > 0):
        ### moved in by one to save time
        acc_test = self.eval(test_loader, save_model=True)
        acc_train = self.eval(
            train_loader, save_model=False, train_fix=True
        )  # Using train acc to pick best model if save model set to True
        self.writer.add_scalar("Acc/training", acc_train, self.curr_epoch)
        og_mem = self.model.dropout.p_mem
        self.model.dropout.p_mem = 1.0
        acc_test_full = self.eval(test_loader, save_model=False, no_manip=True)
        self.model.dropout.p_mem = og_mem
        delta_time = time.process_time() - time_start
        self.save_files["train_time_taken"][0] += delta_time
        self.writer.add_scalar("Loss/train", avg_loss, self.curr_epoch)
        self.writer.add_scalar("Acc/test", acc_test, self.curr_epoch)
        self.writer.add_scalar(
            "Acc/test_no_rescaling", acc_test_full, self.curr_epoch
        )
        self.writer.add_scalar("Acc/train", acc_top, self.curr_epoch)
      self.curr_epoch += 1
    # self.writer.flush()
    return

  def get_save_prefix(self):
    """Constructs the file prefix for saving unlearning results.

    The prefix is based on the pretrain file path, unlearning method, and
    experiment name from the options. If the unlearn method is not "Naive", the
    deletion size is also included in the prefix.
    """
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    if self.opt.unlearn_method != "Naive":
      self.unlearn_file_prefix = (
          self.opt.pretrain_file_prefix
          + "/"
          + str(self.opt.deletion_size)
          + "_"
          + self.opt.unlearn_method
          + "_"
          + self.opt.exp_name
      )
    return

  def compute_and_save_results(
      self,
      train_test_loader,
      test_loader,
      adversarial_train_loader,
      adversarial_test_loader,
      save_model_pth=False,
  ):
    """Computes and saves the results of the unlearning process.

    This method saves the best model (if `save_model_pth` is True),
    training/validation top-1 accuracies, unlearning time, and various
    configuration values. It also evaluates the final model on different loaders
    and saves the predictions and targets.

    Args:
        train_test_loader (DataLoader): DataLoader for the training set used for
          final evaluation.
        test_loader (DataLoader): DataLoader for the test set.
        adversarial_train_loader (DataLoader): DataLoader for adversarial
          examples in the training set.
        adversarial_test_loader (DataLoader): DataLoader for adversarial
          examples in the test set.
        save_model_pth (bool, optional): If True, the best model's state dict is
          saved. Defaults to False.
    """
    print("==> Compute And Save Results In Progress")
    print(
        "------------------#### Dropout mode is: ",
        str(self.model.dropout.drop_mode),
    )
    self.get_save_prefix()
    print(self.unlearn_file_prefix)
    if not exists(self.unlearn_file_prefix):
      makedirs(self.unlearn_file_prefix)
    if save_model_pth:
      torch.save(
          self.best_model.state_dict(), self.unlearn_file_prefix + "/model.pth"
      )
    np.save(
        self.unlearn_file_prefix + "/train_top1.npy",
        self.save_files["train_top1"],
    )
    np.save(
        self.unlearn_file_prefix + "/val_top1.npy", self.save_files["val_top1"]
    )
    np.save(
        self.unlearn_file_prefix + "/unlearn_time.npy",
        self.save_files["train_time_taken"][0],
    )
    # ---------------- NEW ----------------
    # Config values
    for to_log in [
        "inflation",
        "bn_track_running_stats",
        "bn_momentum",
        "pairs",
        "etd_ul",
        "shuffle_mask",
        "ft_end",
        "force_all",
        "train_flip_noise",
        "upscale_mem_during_train",
        "drop_no_scaling",
        "p_scale",
        "tail_comp",
        "rand_dropout_on",
        "rand_dropout_p",
        "rem_mod_potion",
        "rem_mod_ascent",
        "rem_mod_descent",
        "rem_mod_retain",
        "rem_mod_force_df_singlemask",
        "rem_mod_channels",
        "rem_mod_flip_channel_coin",
        "rem_mod_channel_threshold_purge",
        "rem_mod_channel_threshold_channel",
        "run_name",
    ]:
      if to_log == "pairs":
        pairs = getattr(config, to_log)
        to_log_unpacked = ""
        for i in pairs:
          to_log_unpacked += str(i)
        # np.save(self.unlearn_file_prefix + f"/{to_log}.txt", to_log_unpacked)
        # save a s txt, not npy
        with open(self.unlearn_file_prefix + f"/{to_log}.txt", "w") as f:
          f.write(str(to_log_unpacked))
      else:
        log_value = getattr(config, to_log)
        # np.save(self.unlearn_file_prefix + f"/{to_log}.txt", log_value)
        with open(self.unlearn_file_prefix + f"/{to_log}.txt", "w") as f:
          f.write(str(log_value))
    with open(self.unlearn_file_prefix + "/seed.txt", "w") as f:
      f.write(str(self.opt.seed))
    # ----------------
    # self.model = self.best_model.to(device)
    self.model = self.best_model.to(device)  # MAC
    print(
        "==> Completed! Unlearning Time: [{0:.3f}]\t".format(
            self.save_files["train_time_taken"][0]
        )
    )
    for loader, name in [
        (train_test_loader, "train"),
        (test_loader, "test"),
        (adversarial_train_loader, "adv_train"),
        (adversarial_test_loader, "adv_test"),
    ]:
      if loader is not None:
        preds, targets = self.eval(loader=loader, save_preds=True)
        np.save(self.unlearn_file_prefix + "/preds_" + name + ".npy", preds)
        np.save(self.unlearn_file_prefix + "/targets" + name + ".npy", targets)
    return


class ApplyK(Naive):
  """Base class for unlearning methods that apply modifications to a specific part of the model.

  This class extends `Naive` and introduces the concept of applying unlearning
  or modifications to only a subset of the model's layers, determined by the
  parameter `k`. The model is divided into a `prenet` (the frozen part) and the
  `model` (the part being fine-tuned or modified).
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    """Initializes the ApplyK unlearning method.

    Args:
        opt (object): Options object containing hyperparameters, including `k`
          for the layer division, `model_name`, `unlearn_method`, `factor`, and
          `device`.
        model (nn.Module): The full model to be potentially divided.
        prenet (nn.Module, optional): An optional pretrained feature extractor.
          Defaults to None.
        log_name (str, optional): A name for logging purposes. Defaults to
          "None".
    """
    super().__init__(opt, model, prenet, train_type="UL", log_name=log_name)

  def set_model(self, model, prenet):
    """Divides the model based on the `k` parameter and applies unlearning function.

    The model is split into a `prenet` and a trainable `model` using
    `divide_model`. The trainable `model` part is then potentially modified by
    the `unlearn_func`.

    Args:
        model (nn.Module): The original full model.
        prenet (nn.Module): The prenet component, if any.
    """
    prenet, model = self.divide_model(
        model, k=self.opt.k, model_name=self.opt.model.lower()
    )
    model = unlearn_func(
        model=model,
        method=self.opt.unlearn_method,
        factor=self.opt.factor,
        device_=self.opt.device,
    )
    self.model = model
    self.prenet = prenet
    # self.model.to(device)
    self.model.to(device)
    if self.prenet is not None:
      # self.prenet.to(device).eval()
      self.prenet.to(device).eval()

  def divide_model(self, model, k, model_name):
    """Divides the model into a prenet and a trainable network based on `k`.

    The division point depends on the `model_name` and the value of `k`. `k=-1`
    means the entire model is trainable. Specific logic is implemented for
    'resnet9', 'resnetwide28x10', and 'vitb16'.

    Args:
        model (nn.Module): The original model.
        k (int): The parameter determining the division point.
        model_name (str): The name of the model architecture.

    Returns:
        tuple: A tuple containing the prenet (frozen part) and the net
          (trainable part).
    """
    net = None
    prenet = None
    if k == -1:  # -1 means retrain all layers
      net = model
      prenet = None
      return prenet, net
    if model_name == "resnet9":
      assert k in [1, 2, 4, 5, 7, 8]
      mapping = {1: 6, 2: 5, 4: 4, 5: 3, 7: 2, 8: 1}
      dividing_part = mapping[k]
      all_mods = [
          model.conv1,
          model.conv2,
          model.res1,
          model.conv3,
          model.res2,
          model.conv4,
          model.fc,
      ]
      prenet = torch.nn.Sequential(*all_mods[:dividing_part])
      net = torch.nn.Sequential(*all_mods[dividing_part:])
    elif model_name == "resnetwide28x10":
      assert k in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
      all_mods = [
          model.conv1,
          model.layer1,
          model.layer2,
          model.layer3,
          model.norm,
          model.fc,
      ]
      mapping = {1: 5, 9: 3, 17: 2, 25: 1}
      if k in mapping:
        intervention_point = mapping[k]
        prenet = torch.nn.Sequential(*all_mods[:intervention_point])
        net = torch.nn.Sequential(*all_mods[intervention_point:])
      else:
        vals = list(mapping.keys())
        sel_idx = max(vals)  # Initialize sel_idx with the largest key
        for val in vals:
          if val > k:
            sel_idx = val
            break
        layer = mapping[sel_idx]
        prenet_list = all_mods[:layer]
        prenet_additions = list(
            all_mods[layer][: int(4 - (((k - 1) // 2) % 4))]
        )
        prenet = torch.nn.Sequential(*(prenet_list + prenet_additions))
        net_list = list(all_mods[layer][int(4 - (((k - 1) // 2) % 4)) :])
        net_additions = all_mods[layer + 1 :]
        net = torch.nn.Sequential(*(net_list + net_additions))
    elif model_name == "vitb16":
      assert k in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
      all_mods = [model.patch_embed, model.blocks, model.norm, model.head]
      mapping = {1: 3, 13: 1}
      if k in mapping:
        intervention_point = mapping[k]
        prenet = torch.nn.Sequential(*all_mods[:intervention_point])
        net = torch.nn.Sequential(*all_mods[intervention_point:])
      else:
        prenet = [model.patch_embed]
        k = 13 - k
        prenet += [model.blocks[:k]]
        prenet = torch.nn.Sequential(*prenet)
        net = [model.blocks[k:], model.norm, model.head]
        net = torch.nn.Sequential(*net)
    prenet.to(self.opt.device)
    net.to(self.opt.device)
    return prenet, net

  def get_save_prefix(self):
    """Constructs the file prefix for saving results for ApplyK methods.

    The prefix includes the pretrain file path, deletion size, unlearn method,
    experiment name, training iterations, and the `k` parameter.
    """
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    return


class Scrub(ApplyK):
  """Implements the Scrub unlearning method.

  This method combines a phase of maximizing the loss on the forget set with a
  knowledge distillation loss from the original model when training on the
  retain set. It aims to remove information about the forget set while
  preserving knowledge from the original model.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    print(log_name)
    super().__init__(opt, model, prenet, log_name=log_name)
    self.og_model = copy.deepcopy(model)
    # self.og_model.to(device).eval()
    self.og_model.to(device).eval()  # MAC
    self.curr_epoch = 0  # Placeholder only

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    if (
        self.prenet is not None
    ):  # pretrained models are not relevant as we do the pretraining.
      # thus not adapted to work with new additions
      with torch.no_grad():
        feats = self.prenet(images)
      output = self.model(feats)
    else:
      output = self.model(images, idx, epoch=None, classidx=target)
    with torch.no_grad():
      logit_t = self.og_model(images, idx)
    loss = F.cross_entropy(output, target)
    loss += self.opt.alpha * distill_kl_loss(output, logit_t, self.opt.kd_T)
    if self.maximize:
      loss = -loss
    self.top1(output, target)
    return loss

  def unlearn(
      self, train_loader, test_loader, forget_loader, eval_loaders=None
  ):
    self.maximize = False
    while self.curr_step < self.opt.train_iters:
      if self.curr_step < self.opt.msteps:
        self.maximize = True
        time_start = time.process_time()
        self.train_one_epoch(loader=forget_loader)
        delta_time = time.process_time() - time_start
        self.save_files["train_time_taken"][0] += delta_time
        self.eval(loader=test_loader)
      self.maximize = False
      time_start = time.process_time()
      self.train_one_epoch(loader=train_loader)
      delta_time = time.process_time() - time_start
      self.save_files["train_time_taken"][0] += delta_time
      self.eval(loader=test_loader)
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += (
        "_"
        + str(self.opt.kd_T)
        + "_"
        + str(self.opt.alpha)
        + "_"
        + str(self.opt.msteps)
    )
    return


class BadT(ApplyK):
  """Implements the BadT unlearning method.

  This method uses a form of knowledge distillation where the student model is
  trained to mimic a "teacher" ensemble. The teacher's output is a combination
  of the original, fully trained model (`self.og_model`) and a randomly
  initialized model (`self.random_model`). For samples in the forget set
  (indicated by `infgt == 1`), the student distills from the random model. For
  retain samples (`infgt == 0`), it distills from the original model. This
  encourages the model to "forget" the information associated with the forget
  set by aligning with a random baseline, while preserving knowledge on the
  retain set by aligning with the original model.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)
    self.og_model = copy.deepcopy(model)
    self.og_model.to(self.opt.device)
    self.og_model.eval()
    self.random_model = unlearn_func(model, "EU")
    self.random_model.eval()
    self.kltemp = 1

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Performs a forward pass and computes the BadT unlearning loss.

    The loss is based on knowledge distillation. For samples in the forget set
    (`infgt == 1`), the model is distilled from a randomly initialized model.
    For retain samples (`infgt == 0`), it's distilled from the original, fully
    trained model.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor): A binary tensor where 1 indicates a sample to
          forget and 0 indicates a sample to retain. Defaults to None.
        flipsign (bool, optional): Unused in this method. Defaults to False.

    Returns:
        torch.Tensor: A scalar tensor representing the KL divergence loss.
    """
    if self.prenet is not None:
      with torch.no_grad():
        feats = self.prenet(images)
      output = self.model(feats)
    else:
      output = self.model(images, idx)
    full_teacher_logits = self.og_model(images, idx)
    unlearn_teacher_logits = self.random_model(images, idx)
    f_teacher_out = torch.nn.functional.softmax(
        full_teacher_logits / self.kltemp, dim=1
    )
    u_teacher_out = torch.nn.functional.softmax(
        unlearn_teacher_logits / self.kltemp, dim=1
    )
    labels = torch.unsqueeze(infgt, 1)
    overall_teacher_out = labels * u_teacher_out + (1 - labels) * f_teacher_out
    student_out = F.log_softmax(output / self.kltemp, dim=1)
    loss = F.kl_div(student_out, overall_teacher_out)
    self.top1(output, target)
    return loss

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    return


class SalUn(ApplyK):
  """An implementation of the Saliency Unlearning (SaLun) method.

  This method fine-tunes a model by applying a standard loss to the data to be
  retained, while adding a saliency-based penalty term for the data to be
  forgotten. This penalty discourages the model from relying on the input
  features of the forget samples, thereby achieving "unlearning".
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    """Initializes the SaLun unlearning method.

    Args:
        opt (object): Options object containing hyperparameters. Expected to
          have 'alpha' for the saliency penalty weight.
        model (nn.Module): The model to be unlearned.
        prenet (nn.Module, optional): An optional pretrained feature extractor.
          Defaults to None.
        log_name (str, optional): A name for logging purposes. Defaults to
          "None".
    """
    super().__init__(opt, model, prenet, log_name=log_name)
    # The hyperparameter that weights the saliency penalty term.
    self.alpha = getattr(opt, "alpha", 1.0)
    # The standard classification criterion used for the retain set.
    self.criterion = nn.CrossEntropyLoss()

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Performs a forward pass and computes the SaLun loss.

    The total loss is a combination of the cross-entropy loss on the retain set
    and a saliency penalty on the forget set.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor): A binary tensor where 1 indicates a sample to
          forget and 0 indicates a sample to retain.
        flipsign (bool, optional): Unused in this method. Defaults to False.

    Returns:
        torch.Tensor: A scalar tensor representing the total loss for the batch.
    """
    # Gradients w.r.t. inputs are required to compute saliency.
    images.requires_grad = True
    # A single forward pass is performed for the entire batch.
    output = self.model(images, idx)
    # Indices are used to split data into retain and forget sets.
    retain_indices = (infgt == 0).nonzero(as_tuple=True)[0]
    forget_indices = (infgt == 1).nonzero(as_tuple=True)[0]
    total_loss = 0.0
    # 1. Compute Cross-Entropy Loss for the Retain Set
    if retain_indices.numel() > 0:
      output_r, target_r = output[retain_indices], target[retain_indices]
      loss_retain = self.criterion(output_r, target_r)
      total_loss += loss_retain
    # 2. Compute Saliency Penalty for the Forget Set
    if forget_indices.numel() > 0:
      images_f, output_f, target_f = (
          images[forget_indices],
          output[forget_indices],
          target[forget_indices],
      )
      # A temporary loss is computed on forget samples to derive gradients.
      loss_f_ce = self.criterion(output_f, target_f)
      # To get the gradient of the forget loss w.r.t the forget inputs,
      # we compute the gradient w.r.t the whole input batch and then select
      # the relevant parts. `allow_unused=True` is needed because the retain
      # samples in `images` are not part of the `loss_f_ce` computation graph.
      (full_grad,) = torch.autograd.grad(
          loss_f_ce, images, retain_graph=True, allow_unused=True
      )
      # Select gradients corresponding to the forget samples.
      grad_f = full_grad[forget_indices]
      # Saliency is the dot product of an input and its gradient, summed over
      # all feature dimensions to get a per-sample score.
      saliency_map = torch.sum(
          grad_f * images_f, dim=tuple(range(1, images_f.dim()))
      )
      # The penalty is the mean saliency. Adding it to the loss penalizes
      # reliance on the forget samples' features.
      loss_saliency = torch.mean(saliency_map)
      total_loss += self.alpha * loss_saliency
    # Deactivate gradient requirement for images after saliency calculation.
    images.requires_grad = False
    # Calculate accuracy over the entire batch for monitoring performance.
    self.top1(output, target)
    return total_loss

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += "_" + str(self.opt.train_iters)
    return


class NoUnlearning(ApplyK):
  # does nothing
  pass


class SSD(ApplyK):
  """Implements the Selective Synaptic Dampending (SSD) unlearning method.

  This method aims to unlearn a forget set by selectively dampening model
  parameters based on their importance and sensitivity to the forget set, using
  the `ssd_tuning` utility.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def unlearn(
      self, train_loader, test_loader, forget_loader, eval_loaders=None
  ):
    self.opt.train_iters = len(train_loader) + len(forget_loader)
    time_start = time.process_time()
    self.best_model = ssd_tuning(
        self.model,
        forget_loader,
        self.opt.SSDdampening,
        self.opt.SSDselectwt,
        train_loader,
        self.opt.device,
    )
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.SSDdampening) + "_" + str(self.opt.SSDselectwt)
    )
    return


class ASSD(ApplyK):
  """Implements an Adaptive Selective Synaptic Dampening (ASSD) unlearning method.

  This method extends the SSD approach by adaptively searching for a suitable
  dampening fraction (`frac_dl`). It iteratively calls `assd_tuning`, adjusting
  `frac_dl` based on the model's accuracy on the forget set, until a minimum
  accuracy threshold (`min_acc_val`) is reached.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    computed_acc = inputl_model_acc.compute()
    if computed_acc is not None:
      inputl_model_acc = float(computed_acc)  # .item()
    else:
      print("Warning: inputl_model_acc.compute() returned None. Returning 0.0")
      inputl_model_acc = 0.0
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
  ):
    """Performs unlearning using Adaptive Selective Synaptic Dampening (ASSD).

    This method iteratively adjusts the dampening fraction (`frac_dl`) to reduce
    the model's accuracy on the `forget_loader` to a target level, while
    preserving performance on the `train_loader`.

    Args:
        train_loader (DataLoader): DataLoader for the retain set.
        test_loader (DataLoader): DataLoader for the test set.
        forget_loader (DataLoader): DataLoader for the forget set.
        eval_loaders (dict, optional): Additional loaders for evaluation.
          Defaults to None.
        frac_dl (float, optional): The initial fraction for dampening. Defaults
          to None.
        min_acc_val (float, optional): The minimum accuracy value (as a fraction
          of the original forget accuracy) to reach on the forget set. Defaults
          to None.
    """
    time_start = time.process_time()
    if ITERATIVE_SEARCH:
      # -----------------------------
      # start the frac iterations
      original_model = copy.deepcopy(
          self.model
      )  # do not forget to pass to device after reassigning
      max_tries = MAX_TRY
      min_acc = min_acc_val  # MIN_ACC
      # calculate the accuracy of the model on the train_loader
      org_model_acc = self.get_acc(original_model, forget_loader)
      # Remove the old files
      try:
        os.remove(file_name_1)
        os.remove(file_name_2)
      except FileNotFoundError:
        print("No previous importance files")
      for loop_i in range(max_tries):
        loop_start_t = time.process_time()
        self.best_model = assd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,
            self.opt.device,
            frac_dl,
        )
        # calculate the accuracy of the model on the train_loader
        new_model_acc = self.get_acc(self.model, forget_loader)
        # check if minimum acc reached
        if new_model_acc <= min_acc * org_model_acc:
          print("minimum accuracy reached")
          break
        else:
          frac_dl = STEP_MULT * frac_dl
          self.model = copy.deepcopy(original_model)
          print("retry with new frac_dl: ", frac_dl)
        loop_end_t = time.process_time() - loop_start_t
        print(loop_i, " loop time: ", loop_end_t)
      self.save_files["train_time_taken"][0] += time.process_time() - time_start
    else:
      self.opt.train_iters = len(train_loader) + len(forget_loader)
      time_start = time.process_time()
      self.best_model = assd_tuning(
          self.model,
          forget_loader,
          None,
          None,
          train_loader,
          self.opt.device,
          frac_dl,
      )
      self.save_files["train_time_taken"][0] += time.process_time() - time_start
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += "_adaptive"
    return


class ALFSSD(ApplyK):
  """Implements an Adaptive LF SSD unlearning method.

  This method builds upon the loss-free unlearning paper, itself an extension of
  the SSD method.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
  ):
    print("--- Unlearning with ALFSSD ---")
    self.opt.train_iters = len(train_loader) + len(forget_loader)
    time_start = time.process_time()
    # -----------------------------
    # start the frac iterations
    original_model = copy.deepcopy(
        self.model
    )  # do not forget to pass to device after reassigning
    max_tries = MAX_TRY
    min_acc = min_acc_val  # MIN_ACC
    # calculate the accuracy of the model on the train_loader
    org_model_acc = self.get_acc(original_model, forget_loader)
    # Remove the old files
    try:
      os.remove(file_name_1)
      os.remove(file_name_2)
    except FileNotFoundError:
      print("No previous importance files")
    for _ in range(max_tries):
      self.best_model = alfssd_tuning(
          self.model,
          forget_loader,
          None,
          None,
          train_loader,  # train_loader,
          self.opt.device,
          frac_dl,
          train_loader,
      )
      # calculate the accuracy of the model on the train_loader
      new_model_acc = self.get_acc(self.model, forget_loader)
      # check if minimum acc reached
      if new_model_acc <= min_acc * org_model_acc:
        print("minimum accuracy reached")
        break
      else:
        frac_dl = STEP_MULT * frac_dl
        self.model = copy.deepcopy(original_model)
        print("retry with new frac_dl: ", frac_dl)
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += "_adaptivelf"
    return


class XALFSSD(ApplyK):
  """Implements the Potion method, also known as XALFSSD.

  This method is an extension of the ALFSSD, incorporating
  modifications/alternative calculations for better performance
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
  ):
    print("--- Unlearning with XALFSSD ---")
    self.opt.train_iters = len(train_loader) + len(forget_loader)
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    if ITERATIVE_SEARCH:  # iterative
      # -----------------------------
      # start the frac iterations
      original_model = copy.deepcopy(
          self.model
      )  # do not forget to pass to device after reassigning
      max_tries = MAX_TRY
      min_acc = min_acc_val  # MIN_ACC
      # calculate the accuracy of the model on the train_loader
      org_model_acc = self.get_acc(original_model, forget_loader)
      # Remove the old files
      try:
        os.remove(file_name_1)
        os.remove(file_name_2)
      except FileNotFoundError:
        print("No previous importance files")
      original_importances, sample_importances = None, None
      for try_i in range(max_tries):
        print("----------------> Attempt #", try_i)
        self.best_model, original_importances, sample_importances = (
            alfssd_tuning(
                self.model,
                forget_loader,
                None,
                None,
                train_loader,  # train_loader,
                self.opt.device,
                frac_dl,
                clean_retain_loader,
                x_d=True,  # turn on alternative D calc
                original_importances=original_importances,
                sample_importances=sample_importances,
            )
        )
        # calculate the accuracy of the model on the train_loader
        new_model_acc = self.get_acc(self.model, forget_loader)
        # train_new_model_acc = self.get_acc(self.model, train_loader)
        # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
        # check if minimum acc reached
        if new_model_acc <= min_acc * org_model_acc:
          print("minimum accuracy reached")
          break
        elif (
            new_model_acc <= min_acc
        ):  # in case of already being good enough at the start
          print("minimum accuracy reached")
          break
        else:
          frac_dl = STEP_MULT * frac_dl
          self.model = copy.deepcopy(original_model)
          print("retry with new frac_dl: ", frac_dl)
    else:
      # --------
      self.best_model = alfssd_tuning(
          self.model,
          forget_loader,
          None,
          None,
          train_loader,  # train_loader,
          self.opt.device,
          frac_dl,
          clean_retain_loader,
          x_d=True,  # turn on alternative D calc
      )
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


# -------- REMMOD --------------------
class REMMOD(ApplyK):
  """Implements a varian of REM (used for ablations).

  (Ablation: uses ascent instead of NPO)
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def freeze_neurons(self, model, percentage):
    """Freezes a percentage of neurons in Linear and Conv2d layers.

    This method iterates through all modules in the provided model. For each
    `nn.Linear` or `nn.Conv2d` layer, it identifies a specified percentage of
    neurons (based on the first dimension of the weight tensor) and sets their
    corresponding weights to zero during the forward pass using a pre-forward
    hook.

    Args:
        model (nn.Module): The PyTorch model to modify.
        percentage (float): The fraction of neurons to freeze (e.g., 0.1 for
          10%).
    """
    for layer in model.modules():
      if isinstance(layer, (nn.Linear, nn.Conv2d)):
        num_neurons = layer.weight.shape[0]
        num_freeze = int(percentage * num_neurons)
        # Create a mask to identify frozen neurons
        mask = torch.ones_like(layer.weight)  # Start with all ones
        mask[:num_freeze, ...] = 0.0  # Set mask to 0 for neurons to freeze
        # Register a buffer to store the mask (not a trainable parameter)
        layer.register_buffer("freeze_mask", mask)

        # Modify the forward pass to apply the mask
        def hook(module, input_):
          # Apply the mask to the weights
          module.weight.data = module.weight.data * module.freeze_mask
          return input_

        layer.register_forward_pre_hook(hook)

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    """Orchestrates the unlearning process for REMMODIDEAL.

    This method implements a variant of the REM unlearning method, designed to
    provide an upper bound by having perfect knowledge of the deletion set. It
    can optionally start with a Potion-like phase. The main loop involves a
    "Flush" phase using gradient ascent on the forget set and a "Repair" phase
    on the retain set. Various metrics are logged throughout the process.

    Args:
        train_loader (DataLoader): DataLoader for the training set (retain set).
        test_loader (DataLoader): DataLoader for the test set.
        forget_loader (DataLoader): DataLoader for the forget set.
        eval_loaders (dict, optional): Dictionary of additional DataLoaders for
          evaluation. Defaults to None.
        frac_dl (float, optional): Dampening fraction used in the optional
          Potion phase. Defaults to None.
        min_acc_val (float, optional): Minimum accuracy threshold for the
          iterative search in the Potion phase. Defaults to None.
        pretrain_loader (DataLoader, optional): DataLoader for the pretraining
          data, used in the "Repair" phase. Defaults to None.
        delete_idx (list, optional): Indices of samples to be forgotten.
          Defaults to None.
        eval_train (DataLoader, optional): DataLoader for evaluating training
          performance. Defaults to None.
    """
    print("--- Unlearning with REMMOD ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    # =======================
    self.best_model = self.model  # init
    # test if better than before but only using what we are given
    keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
    removing = self.eval(forget_loader, save_model=False, train_fix=True)
    if removing < config.removing_threshold:
      removing = config.removing_threshold
    prev_proxy = keeping * (1 - removing)
    self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
    # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    warmup_counter = 0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          # print(purge_iters, purge_acc)
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader, ascent=True, no_scheduler_step=True
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= CHANNEL ===============================
      if self.model.dropout.p_mem != 0:
        for channel in channels:
          self.model.dropout.force_mask_id = channel
          if config.rem_mod_descent:
            # train forget loader onto the new mask
            self.model.dropout.drop_mode = "train_one_ul_mask"
            print(
                f"Channel {channel}, drop mode: ",
                self.model.dropout.drop_mode,
            )
            guide_acc = self.eval(
                forget_loader,
                save_model=False,
                train_fix=True,
                class_0=False,
                ic=False,
                only_infgti=True,
            )
            channel_iters = 0
            while guide_acc < (1 - config.rem_mod_channel_threshold_channel):
              _, _ = self.train_one_epoch(
                  loader=forget_loader,
                  forced_target=None,
                  delete_idx=delete_idx,
                  no_scheduler_step=True,
              )
              channel_iters += 1
              guide_acc = self.eval(
                  forget_loader,
                  save_model=False,
                  train_fix=True,
                  class_0=False,
                  ic=False,
                  only_infgti=True,
              )
              if channel_iters >= config.rem_mod_maxwhile_iters_channel:
                break
            self.writer.add_scalar(
                f"intermediate/channel{channel}", guide_acc, self.curr_epoch
            )
      # ================================================
      # ================= REPAIR ===============================
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = (  # "train_one_ul_mask"
              "train_one_ul_mask"
          )
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader, forced_target=None, delete_idx=delete_idx
        )
        self.writer.add_scalar(
            "intermediate/pretrain_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      warmup_counter += 1
      # self.best_model = self.model
      for eval_mode in ["drop", "train_one_ul_mask"]:
        self.model.dropout.drop_mode = eval_mode
        print(
            "UL SPECIAL drop_mode set back to inference mode (test): ",
            self.model.dropout.drop_mode,
        )
        acc_adv_train = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        self.writer.add_scalar(
            f"UL_{eval_mode}/forget_{eval_mode}", acc_adv_train, self.curr_epoch
        )
        if eval_mode == "drop":
          acc_adv_train = self.eval(
              test_loader,
              save_model=False,
              train_fix=False,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/test_{eval_mode}", acc_adv_train, self.curr_epoch
          )
          if eval_train is not None:
            acc_adv_train = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      # =========================
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
      self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
      self.writer.add_scalar("intermediate/removing", removing, self.curr_epoch)
      if proxy > prev_proxy:
        self.best_model = copy.deepcopy(self.model)
        prev_proxy = proxy
      # ========================
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class REMMODIDEAL(ApplyK):
  """Implements a varian of REM (used for ablations).

  This variant has perfect knowledge of the deletion set and can therefore
  provide an upper bound on the performance of REM. (uses ascent, not NPO)
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def freeze_neurons(self, model, percentage):
    """Freezes a percentage of neurons in Linear and Conv2d layers.

    This method iterates through all modules in the provided model. For each
    `nn.Linear` or `nn.Conv2d` layer, it identifies a specified percentage of
    neurons (based on the first dimension of the weight tensor) and sets their
    corresponding weights to zero during the forward pass using a pre-forward
    hook.

    Args:
        model (nn.Module): The PyTorch model to modify.
        percentage (float): The fraction of neurons to freeze (e.g., 0.1 for
          10%).
    """
    for layer in model.modules():
      if isinstance(layer, (nn.Linear, nn.Conv2d)):
        num_neurons = layer.weight.shape[0]
        num_freeze = int(percentage * num_neurons)
        # Create a mask to identify frozen neurons
        mask = torch.ones_like(layer.weight)  # Start with all ones
        mask[:num_freeze, ...] = 0.0  # Set mask to 0 for neurons to freeze
        # Register a buffer to store the mask (not a trainable parameter)
        layer.register_buffer("freeze_mask", mask)

        # Modify the forward pass to apply the mask
        def hook(module, input_):
          # Apply the mask to the weights
          module.weight.data = module.weight.data * module.freeze_mask
          return input_

        layer.register_forward_pre_hook(hook)

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with REMMOD ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    # =======================
    self.best_model = self.model  # init
    # test if better than before but only using what we are given
    keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
    removing = self.eval(forget_loader, save_model=False, train_fix=True)
    if removing < config.removing_threshold:
      removing = config.removing_threshold
    prev_proxy = keeping * (1 - removing)
    self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
    # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    warmup_counter = 0
    while self.curr_step < self.opt.train_iters:
      # remaining_steps = self.opt.train_iters - self.curr_step
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          # print(purge_iters, purge_acc)
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader, ascent=True, no_scheduler_step=True
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= CHANNEL ===============================
      if self.model.dropout.p_mem != 0:
        for channel in channels:
          self.model.dropout.force_mask_id = channel
          if config.rem_mod_descent:
            # train forget loader onto the new mask
            self.model.dropout.drop_mode = "train_one_ul_mask"
            print(
                f"Channel {channel}, drop mode: ",
                self.model.dropout.drop_mode,
            )
            guide_acc = self.eval(
                forget_loader,
                save_model=False,
                train_fix=True,
                class_0=False,
                ic=False,
                only_infgti=True,
            )
            channel_iters = 0
            while guide_acc < (1 - config.rem_mod_channel_threshold_channel):
              _, _ = self.train_one_epoch(
                  loader=forget_loader,
                  forced_target=None,
                  delete_idx=delete_idx,
                  no_scheduler_step=True,
              )
              channel_iters += 1
              guide_acc = self.eval(
                  forget_loader,
                  save_model=False,
                  train_fix=True,
                  class_0=False,
                  ic=False,
                  only_infgti=True,
              )
              if channel_iters >= config.rem_mod_maxwhile_iters_channel:
                break
            self.writer.add_scalar(
                f"intermediate/channel{channel}", guide_acc, self.curr_epoch
            )
      # ================================================
      # ================= REPAIR ===============================
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = (  # "train_one_ul_mask"
              "train_one_ul_mask"
          )
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader, forced_target=None, delete_idx=delete_idx
        )
        self.writer.add_scalar(
            "intermediate/pretrain_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      warmup_counter += 1
      # self.best_model = self.model
      for eval_mode in ["drop", "train_one_ul_mask"]:
        self.model.dropout.drop_mode = eval_mode
        print(
            "UL SPECIAL drop_mode set back to inference mode (test): ",
            self.model.dropout.drop_mode,
        )
        acc_adv_train = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        self.writer.add_scalar(
            f"UL_{eval_mode}/forget_{eval_mode}", acc_adv_train, self.curr_epoch
        )
        if eval_mode == "drop":
          acc_adv_train = self.eval(
              test_loader,
              save_model=False,
              train_fix=False,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/test_{eval_mode}", acc_adv_train, self.curr_epoch
          )
          if eval_train is not None:
            acc_adv_train = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      # =========================
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
      self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
      self.writer.add_scalar("intermediate/removing", removing, self.curr_epoch)
      if proxy > prev_proxy:
        self.best_model = copy.deepcopy(self.model)
        prev_proxy = proxy
      # ========================
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class Ascent(ApplyK):
  """Implements an unlearning method using gradient ascent.

  This method aims to reduce the model's performance on the forget set by
  performing gradient ascent on the loss calculated from the forget set. It
  includes evaluation steps to monitor the progress of unlearning.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with ASCENT ---")
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    self.curr_epoch = 50
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    for eval_mode in ["drop"]:
      self.model.dropout.drop_mode = eval_mode
      print(
          "UL SPECIAL drop_mode set back to inference mode (test): ",
          self.model.dropout.drop_mode,
      )
      acc_adv_train = self.eval(
          forget_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=True,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/forget_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      acc_adv_train = self.eval(
          clean_retain_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=False,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/retain_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      acc_adv_train = self.eval(
          test_loader,
          save_model=False,
          train_fix=False,
          class_0=False,
          ic=False,
          only_infgti=False,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/test_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      if eval_train is not None:
        acc_adv_train = self.eval(
            eval_train,
            save_model=False,
            train_fix=False,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        self.writer.add_scalar(
            f"UL_{eval_mode}/healed_{eval_mode}", acc_adv_train, self.curr_epoch
        )
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    # warmup_counter = 0
    prev_proxy = 0
    loop_i = 0
    breaker = 0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      self.model.dropout.drop_mode = "drop"
      print("Ascent with drop mode: ", self.model.dropout.drop_mode)
      purge_acc = self.eval(
          forget_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=True,
      )
      if loop_i == 0:
        self.best_model = copy.deepcopy(self.model)
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          print("### Breaker loop activated as acc below desired already")
          breaker = 1
          purge_acc = 100  # to ensure at least one step.
          # The best model save ensures this does no harm if it is worse
        loop_i += 1
      if purge_acc > config.rem_mod_channel_threshold_purge:
        _, _ = self.train_one_epoch(
            loader=forget_loader, ascent=True, no_scheduler_step=True
        )
        for eval_mode in ["drop"]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_1 = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}", acc_1, self.curr_epoch
          )
          acc_2 = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}", acc_2, self.curr_epoch
          )
          acc_3 = self.eval(
              pretrain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/pretrain_manip_overfit_{eval_mode}",
              acc_3,
              self.curr_epoch,
          )
          acc_4 = self.eval(
              test_loader,
              save_model=False,
              train_fix=False,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/test_{eval_mode}", acc_4, self.curr_epoch
          )
          if eval_train is not None:
            acc_5 = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}", acc_5, self.curr_epoch
            )
      else:
        print("Min acc reached")
        for eval_mode in ["drop"]:
          self.model.dropout.drop_mode = eval_mode
          if eval_train is not None:
            acc_5 = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}", acc_5, self.curr_epoch
            )
        break
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      if breaker == 1:
        break
      # =========================
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
      if proxy > prev_proxy:
        self.best_model = copy.deepcopy(self.model)
        prev_proxy = proxy
      # ========================
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class REMGEN(ApplyK):
  """Implements a varian of REM (used for ablations).

  This variant does not use memorization neurons to show that they are essential
  for performance of REM (Ablation: does not use redirection/mem part of step
  3.1)
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def npo_loss(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Calculates the Negative Preference Optimization (NPO) loss.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor, optional): Binary tensor indicating forget samples.
          Defaults to None.
        flipsign (bool, optional): Flag for loss modification. Defaults to
          False.

    Returns:
        torch.Tensor: The scalar NPO loss.
    """
    _ = epoch
    _ = infgt
    _ = flipsign
    self.beta = 1
    # self.model.dropout.drop_mode = "drop"
    # self.og_model.dropout.drop_mode = "drop"
    # https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py
    output_new = self.model(images, idx, self.curr_epoch, target)
    forget_loss_current = F.cross_entropy(output_new, target, reduction="none")
    with torch.no_grad():
      output_og = self.og_model(images, idx, self.curr_epoch, target)
      forget_loss_oracle = F.cross_entropy(output_og, target, reduction="none")
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
    return loss

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    self.beta = 1
    loss = 0.0
    loss_channel = torch.zeros_like(target, dtype=torch.float32, device=device)
    # check if model in train mode else do soemthing else
    if self.model.training:
      # self.model.dropout.drop_mode = "drop"
      # output_drop = self.model(images, idx, self.curr_epoch, target)
      # loss_drop = F.cross_entropy(output_drop, target, reduction='none')
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      # loss_drop = self.npo_loss(self, images, target, idx)
      # ------------
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      self.og_model.eval()
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
        self.og_model.zero_grad()  # for safety
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      loss_drop = F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
      self.og_model.zero_grad()  # for safety
      # ------------
      if flipsign == "Repair":
        self.model.dropout.drop_mode = "drop"  # train_one_ul_mask
        self.og_model.dropout.drop_mode = "drop"
        # loss_channel = self.npo_loss(self, images, target, idx)
        # ------------
        output_new_rep = self.model(images, idx, self.curr_epoch, target)
        forget_loss_current_rep = F.cross_entropy(
            output_new_rep, target, reduction="none"
        )
        neg_log_ratios_rep = forget_loss_current_rep - forget_loss_oracle
        loss_channel = (
            F.logsigmoid(self.beta * neg_log_ratios_rep) * 2 / self.beta
        )
        # ------------
      # self.model.dropout.drop_mode = "drop" # setting back just in case
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      if config.inflation != 0:
        inflation = (
            50000 / len(self.model.dropout.drop_targets) * config.inflation
        )
      else:
        inflation = 1
      if flipsign == "Flush":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1  # * inflation #
        # loss = loss_drop.mean()
        loss = torch.mean(loss_drop)
      elif flipsign == "Repair":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1 * inflation  # remove if no purge needed
            loss_channel[i] *= (
                0 * inflation
            )  # using the knowledge about scaling
          else:
            loss_drop[i] *= 0.0
            # loss_channel[i] *= 0.0
        if config.tc_ascent:
          pass
        else:
          loss_drop *= 0.0  # set it zero to only have channeling
        loss = loss_drop + loss_channel
        loss = loss / max(1, inflation)  # avoid explosion
        loss = torch.mean(loss)
      elif flipsign == "ascent":
        raise AssertionError("Ascent is not supported in this forward pass.")
      # self.top1(output_drop, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with REMGEN ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    train_style = "train_one_ul_mask"  # "train_one_ul_mask"
    # prev_proxy = 0.0  # Initialize prev_proxy
    if config.best_model_check:
      # =======================
      self.best_model = self.model  # init
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      prev_proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
      # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    self.og_model = copy.deepcopy(self.model)
    warmup_counter = 0
    prev_proxy = 0.0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      self.no_purge = True
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      print("==> FLUSH")
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          self.no_purge = True
        else:
          self.no_purge = False
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          # print(purge_iters, purge_acc)
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader,
              ascent=False,
              no_scheduler_step=True,
              flipsign="Flush",
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          print(
              "Flush iter #",
              purge_iters,
              " Acc: ",
              purge_acc,
              "Train acc: ",
              acc_top,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= REPAIR ===============================
      print("==> REPAIR AND CHANNEL")
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = train_style  # "train_one_ul_mask"
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader,
            forced_target=None,
            delete_idx=delete_idx,
            flipsign="Repair",
        )
        self.writer.add_scalar(
            "intermediate/trainacc_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      print("==> EVAL")
      warmup_counter += 1
      # self.best_model = self.model
      if config.full_logging:
        for eval_mode in ["drop"]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_adv_train = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          acc_adv_train = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          if eval_mode == "drop":
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=False,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/healed_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
          if eval_mode == train_style:
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=True,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/MASK_poisonworks_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      if config.best_model_check:
        # =========================
        # test if better than before but only using what we are given
        keeping = self.eval(
            clean_retain_loader, save_model=False, train_fix=True
        )
        removing = self.eval(forget_loader, save_model=False, train_fix=True)
        if removing < config.removing_threshold:
          removing = config.removing_threshold
        proxy = keeping * (1 - removing)
        self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
        self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
        self.writer.add_scalar(
            "intermediate/removing", removing, self.curr_epoch
        )
        if proxy > prev_proxy:
          self.best_model = copy.deepcopy(self.model)
          prev_proxy = proxy
        # ========================
      else:
        self.best_model = copy.deepcopy(self.model)
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class NPORT(ApplyK):
  """Implements NPO with retraining after NPO."""

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def npo_loss(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Calculates the Negative Preference Optimization (NPO) loss.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor, optional): Binary tensor indicating forget samples.
          Defaults to None.
        flipsign (bool, optional): Flag for loss modification. Defaults to
          False.

    Returns:
        torch.Tensor: The scalar NPO loss.
    """
    _ = epoch
    _ = infgt
    _ = flipsign
    self.beta = 1
    # self.model.dropout.drop_mode = "drop"
    # self.og_model.dropout.drop_mode = "drop"
    # https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py
    output_new = self.model(images, idx, self.curr_epoch, target)
    forget_loss_current = F.cross_entropy(output_new, target, reduction="none")
    with torch.no_grad():
      output_og = self.og_model(images, idx, self.curr_epoch, target)
      forget_loss_oracle = F.cross_entropy(output_og, target, reduction="none")
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
    return loss

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    self.beta = 1
    loss = 0.0  # Initialize loss
    loss_channel = torch.zeros_like(target, dtype=torch.float32, device=device)
    # to avoid undefined loss
    # check if model in train mode else do soemthing else
    if self.model.training:
      # self.model.dropout.drop_mode = "drop"
      # output_drop = self.model(images, idx, self.curr_epoch, target)
      # loss_drop = F.cross_entropy(output_drop, target, reduction='none')
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "train_one_ul_mask"
      # loss_drop = self.npo_loss(self, images, target, idx)
      # ------------
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      self.og_model.eval()
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
        self.og_model.zero_grad()  # for safety
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      loss_drop = F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
      self.og_model.zero_grad()  # for safety
      # ------------
      if flipsign == "Repair":
        self.model.dropout.drop_mode = "train_one_ul_mask"
        output_drop_channel = self.model(images, idx, self.curr_epoch, target)
        loss_channel = F.cross_entropy(
            output_drop_channel, target, reduction="none"
        )
      # self.model.dropout.drop_mode = "drop" # setting back just in case
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      if config.inflation != 0:
        inflation = (
            50000 / len(self.model.dropout.drop_targets) * config.inflation
        )
      else:
        inflation = 1
      if flipsign == "Flush":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1  # * inflation #
        # loss = loss_drop.mean()
        loss = torch.mean(loss_drop)
      elif flipsign == "Repair":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1 * inflation  # remove if no purge needed
            loss_channel[i] *= (
                0 * inflation
            )  # using the knowledge about scaling
          else:
            loss_drop[i] *= 0.0
            # loss_channel[i] *= 0.0
        if config.tc_ascent:
          pass
        else:
          loss_drop *= 0.0  # set it zero to only have channeling
        loss = loss_drop + loss_channel
        loss = loss / max(1, inflation)  # avoid explosion
        loss = torch.mean(loss)
      elif flipsign == "ascent":
        assert False, "Ascent is not supported in this forward pass."
      # self.top1(output_drop, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with NPORT---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    train_style = "train_one_ul_mask"  # "train_one_ul_mask"
    if config.best_model_check:
      # =======================
      self.best_model = self.model  # init
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      prev_proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
      # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    self.og_model = copy.deepcopy(self.model)
    warmup_counter = 0
    prev_proxy = 0.0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      self.no_purge = True
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      print("==> FLUSH")
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          self.no_purge = True
        else:
          self.no_purge = False
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          # print(purge_iters, purge_acc)
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader,
              ascent=False,
              no_scheduler_step=True,
              flipsign="Flush",
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          print(
              "Flush iter #",
              purge_iters,
              " Acc: ",
              purge_acc,
              "Train acc: ",
              acc_top,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
          # print(purge_iters, purge_acc, test_acci)
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= CHANNEL ===============================
      # ================================================
      # ================= REPAIR ===============================
      print("==> REPAIR AND CHANNEL")
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = train_style  # "train_one_ul_mask"
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader,
            forced_target=None,
            delete_idx=delete_idx,
            flipsign="Repair",
        )
        self.writer.add_scalar(
            "intermediate/trainacc_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      print("==> EVAL")
      warmup_counter += 1
      # self.best_model = self.model
      if config.full_logging:
        for eval_mode in ["drop", train_style]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_adv_train = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          acc_adv_train = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          if eval_mode == "drop":
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=False,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/healed_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
          if eval_mode == train_style:
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=True,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/MASK_poisonworks_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      if config.best_model_check:
        # =========================
        # test if better than before but only using what we are given
        keeping = self.eval(
            clean_retain_loader, save_model=False, train_fix=True
        )
        removing = self.eval(forget_loader, save_model=False, train_fix=True)
        if removing < config.removing_threshold:
          removing = config.removing_threshold
        proxy = keeping * (1 - removing)
        self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
        self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
        self.writer.add_scalar(
            "intermediate/removing", removing, self.curr_epoch
        )
        if proxy > prev_proxy:
          self.best_model = copy.deepcopy(self.model)
          prev_proxy = proxy
        # ========================
      else:
        self.best_model = copy.deepcopy(self.model)
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class REM(ApplyK):
  """Implements REM as described in the paper."""

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def npo_loss(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Calculates the Negative Preference Optimization (NPO) loss.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor, optional): Binary tensor indicating forget samples.
          Defaults to None.
        flipsign (bool, optional): Flag for loss modification. Defaults to
          False.

    Returns:
        torch.Tensor: The scalar NPO loss.
    """
    _ = epoch
    _ = infgt
    _ = flipsign
    self.beta = 1
    # self.model.dropout.drop_mode = "drop"
    # self.og_model.dropout.drop_mode = "drop"
    # https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py
    output_new = self.model(images, idx, self.curr_epoch, target)
    forget_loss_current = F.cross_entropy(output_new, target, reduction="none")
    with torch.no_grad():
      output_og = self.og_model(images, idx, self.curr_epoch, target)
      forget_loss_oracle = F.cross_entropy(output_og, target, reduction="none")
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
    return loss

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    self.beta = 1
    loss = 0.0
    loss_channel = 0.0
    # check if model in train mode else do soemthing else
    if self.model.training:
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "train_one_ul_mask"
      # loss_drop = self.npo_loss(self, images, target, idx)
      # ------------
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      self.og_model.eval()
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
        self.og_model.zero_grad()  # for safety
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      loss_drop = F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
      self.og_model.zero_grad()  # for safety
      # ------------
      if flipsign == "Repair":
        self.model.dropout.drop_mode = "train_one_ul_mask"  # train_one_ul_mask
        self.og_model.dropout.drop_mode = "train_one_ul_mask"
        # loss_channel = self.npo_loss(self, images, target, idx)
        # ------------
        output_new_rep = self.model(images, idx, self.curr_epoch, target)
        forget_loss_current_rep = F.cross_entropy(
            output_new_rep, target, reduction="none"
        )
        neg_log_ratios_rep = forget_loss_current_rep - forget_loss_oracle
        loss_channel = (
            F.logsigmoid(self.beta * neg_log_ratios_rep) * 2 / self.beta
        )
        # ------------
      # self.model.dropout.drop_mode = "drop" # setting back just in case
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      if config.inflation != 0:
        inflation = (
            50000 / len(self.model.dropout.drop_targets) * config.inflation
        )
      else:
        inflation = 1
      if flipsign == "Flush":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1  # * inflation #
        # loss = loss_drop.mean()
        loss = torch.mean(loss_drop)
      elif flipsign == "Repair":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1 * inflation  # remove if no purge needed
            loss_channel[i] *= (
                1 * inflation
            )  # using the knowledge about scaling
          else:
            loss_drop[i] *= 0.0
            # loss_channel[i] *= 0.0
        if config.tc_ascent:
          pass
        else:
          loss_drop *= 0.0  # set it zero to only have channeling
        loss = loss_drop + loss_channel
        loss = loss / max(1, inflation)  # avoid explosion
        loss = torch.mean(loss)
      elif flipsign == "ascent":
        assert False, "Ascent is not supported in this forward pass."
      # self.top1(output_drop, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with REM ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    train_style = "train_one_ul_mask"  # "train_one_ul_mask"
    prev_proxy = 0.0  # linter pleasing
    if config.best_model_check:
      # =======================
      self.best_model = self.model  # init
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      prev_proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
      # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    self.og_model = copy.deepcopy(self.model)
    warmup_counter = 0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      self.no_purge = True
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      print("==> FLUSH")
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          self.no_purge = True
        else:
          self.no_purge = False
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader,
              ascent=False,
              no_scheduler_step=True,
              flipsign="Flush",
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          print(
              "Flush iter #",
              purge_iters,
              " Acc: ",
              purge_acc,
              "Train acc: ",
              acc_top,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
          # print(purge_iters, purge_acc, test_acci)
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= CHANNEL ===============================
      # ================================================
      # ================= REPAIR ===============================
      print("==> REPAIR AND CHANNEL")
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = train_style  # "train_one_ul_mask"
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader,
            forced_target=None,
            delete_idx=delete_idx,
            flipsign="Repair",
        )
        self.writer.add_scalar(
            "intermediate/trainacc_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      print("==> EVAL")
      warmup_counter += 1
      # self.best_model = self.model
      if config.full_logging:
        for eval_mode in ["drop", train_style]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_adv_train = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          acc_adv_train = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          if eval_mode == "drop":
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=False,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/healed_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
          if eval_mode == train_style:
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=True,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/MASK_poisonworks_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      if config.best_model_check:
        # =========================
        # test if better than before but only using what we are given
        keeping = self.eval(
            clean_retain_loader, save_model=False, train_fix=True
        )
        removing = self.eval(forget_loader, save_model=False, train_fix=True)
        if removing < config.removing_threshold:
          removing = config.removing_threshold
        proxy = keeping * (1 - removing)
        self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
        self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
        self.writer.add_scalar(
            "intermediate/removing", removing, self.curr_epoch
        )
        if proxy > prev_proxy:
          self.best_model = copy.deepcopy(self.model)
          prev_proxy = proxy
        # ========================
      else:
        self.best_model = copy.deepcopy(self.model)
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class REMIDEAL(ApplyK):
  """Implements a varian of REM (used for ablations).

  This variant has full knowledge of the corrupted data. Provides an upper bound
  on the performance of REM.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def npo_loss(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Calculates the Negative Preference Optimization (NPO) loss.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor, optional): Binary tensor indicating forget samples.
          Defaults to None.
        flipsign (bool, optional): Flag for loss modification. Defaults to
          False.

    Returns:
        torch.Tensor: The scalar NPO loss.
    """
    _ = epoch
    _ = infgt
    _ = flipsign
    self.beta = 1
    # self.model.dropout.drop_mode = "drop"
    # self.og_model.dropout.drop_mode = "drop"
    # https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py
    output_new = self.model(images, idx, self.curr_epoch, target)
    forget_loss_current = F.cross_entropy(output_new, target, reduction="none")
    with torch.no_grad():
      output_og = self.og_model(images, idx, self.curr_epoch, target)
      forget_loss_oracle = F.cross_entropy(output_og, target, reduction="none")
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
    return loss

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    self.beta = 1
    loss = 0.0
    loss_channel = 0.0
    # check if model in train mode else do soemthing else
    if self.model.training:
      # self.model.dropout.drop_mode = "drop"
      # output_drop = self.model(images, idx, self.curr_epoch, target)
      # loss_drop = F.cross_entropy(output_drop, target, reduction='none')
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "train_one_ul_mask"
      # loss_drop = self.npo_loss(self, images, target, idx)
      # ------------
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      self.og_model.eval()
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
        self.og_model.zero_grad()  # for safety
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      loss_drop = F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
      self.og_model.zero_grad()  # for safety
      # ------------
      if flipsign == "Repair":
        self.model.dropout.drop_mode = "train_one_ul_mask"  # train_one_ul_mask
        self.og_model.dropout.drop_mode = "train_one_ul_mask"
        # loss_channel = self.npo_loss(self, images, target, idx)
        # ------------
        output_new_rep = self.model(images, idx, self.curr_epoch, target)
        forget_loss_current_rep = F.cross_entropy(
            output_new_rep, target, reduction="none"
        )
        neg_log_ratios_rep = forget_loss_current_rep - forget_loss_oracle
        loss_channel = (
            F.logsigmoid(self.beta * neg_log_ratios_rep) * 2 / self.beta
        )
        # ------------
      # self.model.dropout.drop_mode = "drop" # setting back just in case
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      if config.inflation != 0:
        inflation = (
            50000 / len(self.model.dropout.drop_targets) * config.inflation
        )
      else:
        inflation = 1
      if flipsign == "Flush":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1  # * inflation #
        # loss = loss_drop.mean()
        loss = torch.mean(loss_drop)
      elif flipsign == "Repair":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1 * inflation  # remove if no purge needed
            loss_channel[i] *= (
                1 * inflation
            )  # using the knowledge about scaling
          else:
            loss_drop[i] *= 0.0
            # loss_channel[i] *= 0.0
        if config.tc_ascent:
          pass
        else:
          loss_drop *= 0.0  # set it zero to only have channeling
        loss = loss_drop + loss_channel
        loss = loss / max(1, inflation)  # avoid explosion
        loss = torch.mean(loss)
      elif flipsign == "ascent":
        assert False, "Ascent is not supported in this forward pass."
      # self.top1(output_drop, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with REM IDEAL ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    train_style = "train_one_ul_mask"  # "train_one_ul_mask"
    if config.best_model_check:
      # =======================
      self.best_model = self.model  # init
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      prev_proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
      # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    self.og_model = copy.deepcopy(self.model)
    warmup_counter = 0
    prev_proxy = 0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      self.no_purge = True
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      print("==> FLUSH")
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          self.no_purge = True
        else:
          self.no_purge = False
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader,
              ascent=False,
              no_scheduler_step=True,
              flipsign="Flush",
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          print(
              "Flush iter #",
              purge_iters,
              " Acc: ",
              purge_acc,
              "Train acc: ",
              acc_top,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================================================
      # ================= REPAIR ===============================
      print("==> REPAIR AND CHANNEL")
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = train_style  # "train_one_ul_mask"
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader,
            forced_target=None,
            delete_idx=delete_idx,
            flipsign="Repair",
        )
        self.writer.add_scalar(
            "intermediate/trainacc_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      print("==> EVAL")
      warmup_counter += 1
      # self.best_model = self.model
      if config.full_logging:
        for eval_mode in ["drop", train_style]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_adv_train = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          acc_adv_train = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          if eval_mode == "drop":
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=False,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/healed_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
          if eval_mode == train_style:
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=True,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/MASK_poisonworks_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      if config.best_model_check:
        # =========================
        # test if better than before but only using what we are given
        keeping = self.eval(
            clean_retain_loader, save_model=False, train_fix=True
        )
        removing = self.eval(forget_loader, save_model=False, train_fix=True)
        if removing < config.removing_threshold:
          removing = config.removing_threshold
        proxy = keeping * (1 - removing)
        self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
        self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
        self.writer.add_scalar(
            "intermediate/removing", removing, self.curr_epoch
        )
        if proxy > prev_proxy:
          self.best_model = copy.deepcopy(self.model)
          prev_proxy = proxy
        # ========================
      else:
        self.best_model = copy.deepcopy(self.model)
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class NPOREMMOD(ApplyK):
  """Implements a varian of REM (used for ablations).

  (Ablation: no removal during redirection with step 3.2)
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def npo_loss(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    """Calculates the Negative Preference Optimization (NPO) loss.

    Args:
        images (torch.Tensor): A batch of input images.
        target (torch.Tensor): A batch of ground-truth labels.
        idx (torch.Tensor): A batch of sample indices.
        epoch (int, optional): The current epoch. Defaults to None.
        infgt (torch.Tensor, optional): Binary tensor indicating forget samples.
          Defaults to None.
        flipsign (bool, optional): Flag for loss modification. Defaults to
          False.

    Returns:
        torch.Tensor: The scalar NPO loss.
    """
    _ = epoch
    _ = infgt
    _ = flipsign
    self.beta = 1
    # self.model.dropout.drop_mode = "drop"
    # self.og_model.dropout.drop_mode = "drop"
    # https://github.com/licong-lin/negative-preference-optimization/blob/main/TOFU/dataloader.py
    output_new = self.model(images, idx, self.curr_epoch, target)
    forget_loss_current = F.cross_entropy(output_new, target, reduction="none")
    with torch.no_grad():
      output_og = self.og_model(images, idx, self.curr_epoch, target)
      forget_loss_oracle = F.cross_entropy(output_og, target, reduction="none")
    neg_log_ratios = forget_loss_current - forget_loss_oracle
    loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
    return loss

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    self.beta = 1
    loss_channel = 0.0
    loss = 0.0
    # check if model in train mode else do soemthing else
    if self.model.training:
      # self.model.dropout.drop_mode = "drop"
      # output_drop = self.model(images, idx, self.curr_epoch, target)
      # loss_drop = F.cross_entropy(output_drop, target, reduction='none')
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "train_one_ul_mask"
      # loss_drop = self.npo_loss(self, images, target, idx)
      # ------------
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      self.og_model.eval()
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
        self.og_model.zero_grad()  # for safety
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      loss_drop = F.logsigmoid(self.beta * neg_log_ratios) * 2 / self.beta
      self.og_model.zero_grad()  # for safety
      # ------------
      if flipsign == "Repair":
        self.model.dropout.drop_mode = "train_one_ul_mask"  # train_one_ul_mask
        self.og_model.dropout.drop_mode = "train_one_ul_mask"
        # loss_channel = self.npo_loss(self, images, target, idx)
        # ------------
        output_new_rep = self.model(images, idx, self.curr_epoch, target)
        forget_loss_current_rep = F.cross_entropy(
            output_new_rep, target, reduction="none"
        )
        neg_log_ratios_rep = forget_loss_current_rep - forget_loss_oracle
        loss_channel = (
            F.logsigmoid(self.beta * neg_log_ratios_rep) * 2 / self.beta
        )
        # ------------
      # self.model.dropout.drop_mode = "drop" # setting back just in case
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      if config.inflation != 0:
        inflation = (
            50000 / len(self.model.dropout.drop_targets) * config.inflation
        )
      else:
        inflation = 1
      if flipsign == "Flush":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= -1  # * inflation #
        loss = torch.mean(loss_drop)
      elif flipsign == "Repair":
        for i, idxi in enumerate(idx):
          if idxi in self.model.dropout.drop_targets.cuda():
            loss_drop[i] *= 0 * inflation  # remove if no purge needed
            loss_channel[i] *= (
                1 * inflation
            )  # using the knowledge about scaling
          else:
            loss_drop[i] *= 0.0
            # loss_channel[i] *= 0.0
        loss_drop *= 0.0  # set it zero to only have channeling
        loss = loss_channel
        loss = loss / max(1, inflation)  # avoid explosion
        loss = torch.mean(loss)
      elif flipsign == "ascent":
        assert False, "Ascent is not supported in this forward pass."
      # self.top1(output_drop, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    print("--- Unlearning with REM modifications ---")
    print("gen/mem", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    # opt_save = copy.deepcopy(self.optimizer)
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    # -----------------------------
    run_potion = config.rem_mod_potion
    if run_potion:
      if ITERATIVE_SEARCH:  # iterative
        # -----------------------------
        # start the frac iterations
        original_model = copy.deepcopy(
            self.model
        )  # do not forget to pass to device after reassigning
        max_tries = MAX_TRY
        min_acc = min_acc_val  # MIN_ACC
        # calculate the accuracy of the model on the train_loader
        org_model_acc = self.get_acc(original_model, forget_loader)
        # Remove the old files
        try:
          os.remove(file_name_1)
          os.remove(file_name_2)
        except FileNotFoundError:
          print("No previous importance files")
        for try_i in range(max_tries):
          print("----------------> Attempt #", try_i)
          self.best_model = alfssd_tuning(
              self.model,
              forget_loader,
              None,
              None,
              train_loader,  # train_loader,
              self.opt.device,
              frac_dl,
              clean_retain_loader,
              x_d=True,  # turn on alternative D calc
          )
          # calculate the accuracy of the model on the train_loader
          new_model_acc = self.get_acc(self.model, forget_loader)
          # train_new_model_acc = self.get_acc(self.model, train_loader)
          # print(f"Poison: {new_model_acc}, Train: {train_new_model_acc}")
          # check if minimum acc reached
          if new_model_acc <= min_acc * org_model_acc:
            print("minimum accuracy reached")
            break
          elif (
              new_model_acc <= min_acc
          ):  # in case of already being good enough at the start
            print("minimum accuracy reached")
            break
          else:
            frac_dl = STEP_MULT * frac_dl
            self.model = copy.deepcopy(original_model)
            print("retry with new frac_dl: ", frac_dl)
      else:
        # --------
        self.best_model = alfssd_tuning(
            self.model,
            forget_loader,
            None,
            None,
            train_loader,  # train_loader,
            self.opt.device,
            frac_dl,
            clean_retain_loader,
            x_d=True,  # turn on alternative D calc
            optimizer=self.optimizer,
        )
    # ------- FT until original accuracy is reached again
    self.curr_epoch = 50
    train_style = "train_one_ul_mask"  # "train_one_ul_mask"
    prev_proxy = 0.0
    if config.best_model_check:
      # =======================
      self.best_model = self.model  # init
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      prev_proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", prev_proxy, 50)
      # =======================
    if config.force_rem_mod_on_memzero:
      if self.model.dropout.p_mem == 0:
        self.model.dropout.p_mem = config.force_pmem
        print("==> Setting p_mem from 0 to ", self.model.dropout.p_mem)
      if self.model.dropout.p_fixed == 1:
        self.model.dropout.p_fixed = config.force_pgen
        print("==> Setting p_gen from 1 to ", self.model.dropout.p_fixed)
      # aslo reset the masks
      for i in range(self.model.dropout.max_pairs):
        self.model.dropout.mask_ready_dict[i] = torch.zeros(
            self.model.dropout.max_id
        ).to(device)
        self.model.dropout.mask_tensor_dict[i] = None
        self.model.dropout.mask_tensor.append(None)
        self.model.dropout.count_mask_tensor.append(None)
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    # use two independent schedulers
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    self.model.dropout.drop_targets = delete_idx  # discovered idxs
    if self.opt.optim == "ADAM":
      self.optimizer = torch.optim.Adam(
          self.model.parameters(),
          lr=self.opt.max_lr,
          weight_decay=self.opt.wd,
      )
      self.scheduler = None
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    self.og_model = copy.deepcopy(self.model)
    warmup_counter = 0
    while self.curr_step < self.opt.train_iters:
      # remaining_steps = self.opt.train_iters - self.curr_step
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      channels = np.arange(config.rem_mod_channels)
      self.no_purge = True
      acc_top = 1  # just for cases where we skip because finished at start
      # ================= FLUSH ===============================
      print("==> FLUSH")
      if config.rem_mod_ascent:
        self.model.dropout.drop_mode = "drop"
        print("Ascent with drop mode: ", self.model.dropout.drop_mode)
        purge_acc = self.eval(
            forget_loader,
            save_model=False,
            train_fix=True,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          self.no_purge = True
        else:
          self.no_purge = False
        purge_iters = 0
        while purge_acc > config.rem_mod_channel_threshold_purge:
          # print(purge_iters, purge_acc)
          purge_iters += 1
          _, acc_top = self.train_one_epoch(
              loader=forget_loader,
              ascent=False,
              no_scheduler_step=True,
              flipsign="Flush",
          )
          purge_acc = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          print(
              "Flush iter #",
              purge_iters,
              " Acc: ",
              purge_acc,
              "Train acc: ",
              acc_top,
          )
          if purge_iters >= config.rem_mod_maxwhile_iters_flush:
            break
        self.writer.add_scalar(
            "intermediate/forget_ascent", acc_top, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_acc", purge_acc, self.curr_epoch
        )
        self.writer.add_scalar(
            "intermediate/purge_iters", purge_iters, self.curr_epoch
        )
      # ================= REPAIR ===============================
      print("==> REPAIR AND CHANNEL")
      self.model.dropout.force_mask_id = channels[0]
      if config.rem_mod_retain and warmup_counter >= config.rem_mod_warmup:
        if config.prevent_mem:
          self.model.dropout.drop_mode = "drop"
        else:
          self.model.dropout.drop_mode = train_style  # "train_one_ul_mask"
        print("Retain drop mode: ", self.model.dropout.drop_mode)
        _, acc_top = self.train_one_epoch(
            loader=pretrain_loader,
            forced_target=None,
            delete_idx=delete_idx,
            flipsign="Repair",
        )
        self.writer.add_scalar(
            "intermediate/trainacc_trainmode", acc_top, self.curr_epoch
        )
      # ================================================
      print("==> EVAL")
      warmup_counter += 1
      # self.best_model = self.model
      if config.full_logging:
        for eval_mode in ["drop", train_style]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_adv_train = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          acc_adv_train = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}",
              acc_adv_train,
              self.curr_epoch,
          )
          if eval_mode == "drop":
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=False,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/healed_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
          if eval_mode == train_style:
            if eval_train is not None:
              acc_adv_train = self.eval(
                  eval_train,
                  save_model=False,
                  train_fix=False,
                  class_0=True,
                  ic=False,
                  only_infgti=False,
              )
              self.writer.add_scalar(
                  f"UL_{eval_mode}/MASK_poisonworks_{eval_mode}",
                  acc_adv_train,
                  self.curr_epoch,
              )
            acc_adv_train = self.eval(
                test_loader,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/test_{eval_mode}",
                acc_adv_train,
                self.curr_epoch,
            )
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      self.writer.add_scalar(
          "intermediate/p_mem", self.model.dropout.p_mem, self.curr_epoch
      )
      self.writer.add_scalar(
          "intermediate/p_fixed", self.model.dropout.p_fixed, self.curr_epoch
      )
      if config.best_model_check:
        # =========================
        # test if better than before but only using what we are given
        keeping = self.eval(
            clean_retain_loader, save_model=False, train_fix=True
        )
        removing = self.eval(forget_loader, save_model=False, train_fix=True)
        if removing < config.removing_threshold:
          removing = config.removing_threshold
        proxy = keeping * (1 - removing)
        self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
        self.writer.add_scalar("intermediate/keeping", keeping, self.curr_epoch)
        self.writer.add_scalar(
            "intermediate/removing", removing, self.curr_epoch
        )
        if proxy > prev_proxy:
          self.best_model = copy.deepcopy(self.model)
          prev_proxy = proxy
        # ========================
      else:
        self.best_model = copy.deepcopy(self.model)
      self.curr_epoch += 1
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return


class NPO(ApplyK):
  """Implements the Negative Preference Optimization (NPO) unlearning method.

  This method aims to unlearn a forget set by minimizing a loss that encourages
  the current model's predictions on the forget set to be less confident than
  those of an original, fully trained model. This is achieved by using a loss
  derived from the difference in cross-entropy between the current and the
  original model on the forget set.
  """

  def __init__(self, opt, model, prenet=None, log_name="None"):
    super().__init__(opt, model, prenet, log_name=log_name)

  def get_acc(self, input_model, input_loader):
    """Calculates the top-1 accuracy of a model on a given data loader.

    Args:
        input_model (nn.Module): The model for which to calculate accuracy.
        input_loader (DataLoader): The data loader containing the evaluation
          data.

    Returns:
        float: The top-1 accuracy of the model on the loader.
    """
    # calculate the accuracy of model on the loader
    input_model.eval()
    input_model.to(device)
    inputl_model_acc = torchmetrics.Accuracy(
        task="multiclass", num_classes=self.opt.num_classes
    ).to(device)
    inputl_model_acc.reset()
    with torch.no_grad():
      for images, target, _, _ in input_loader:
        images, target = images.to(device), target.to(device)
        output = input_model(images, torch.zeros_like(target, device=device))
        inputl_model_acc(output, target)
    inputl_model_acc = float(inputl_model_acc.compute())  # .item()
    print("input_model_accuracy: ", inputl_model_acc)
    return inputl_model_acc

  def forward_pass(
      self, images, target, idx, epoch=None, infgt=None, flipsign=False
  ):
    # check if model in train mode else do soemthing else
    if self.model.training:
      self.beta = 1
      self.model.dropout.drop_mode = "drop"
      self.og_model.dropout.drop_mode = "drop"
      # Original from https://github.com/licong-lin/...
      # ...negative-preference-optimization/blob/main/TOFU/dataloader.py
      output_new = self.model(images, idx, self.curr_epoch, target)
      forget_loss_current = F.cross_entropy(
          output_new, target, reduction="none"
      )
      with torch.no_grad():
        output_og = self.og_model(images, idx, self.curr_epoch, target)
        forget_loss_oracle = F.cross_entropy(
            output_og, target, reduction="none"
        )
      neg_log_ratios = forget_loss_current - forget_loss_oracle
      # loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
      # taking out (-) as we apply that via ascent modifier to the returned loss
      loss = F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta
      loss = torch.mean(loss)
      self.top1(output_new, target)
      return loss
    else:
      print("=========== IN INFERENCE MODE =========")
      output = self.model(images, idx, self.curr_epoch, target)
      loss = F.cross_entropy(output, target)
      self.top1(output, target)
      return loss

  def unlearn(
      self,
      train_loader,
      test_loader,
      forget_loader,
      eval_loaders=None,
      frac_dl=None,
      min_acc_val=None,
      pretrain_loader=None,
      delete_idx=None,
      eval_train=None,
  ):
    """Performs unlearning using the Negative Preference Optimization (NPO) method.

    Args:
        train_loader (DataLoader): DataLoader for the training set (retain set).
        test_loader (DataLoader): DataLoader for the test set.
        forget_loader (DataLoader): DataLoader for the forget set.
        eval_loaders (dict, optional): Dictionary of additional DataLoaders for
          evaluation. Defaults to None.
        frac_dl (float, optional): Dampening fraction used in the optional
          Potion phase. Defaults to None.
        min_acc_val (float, optional): Minimum accuracy threshold for the
          iterative search in the Potion phase. Defaults to None.
        pretrain_loader (DataLoader, optional): DataLoader for the pretraining
          data, used in the "Repair" phase. Defaults to None.
        delete_idx (list, optional): Indices of samples to be forgotten.
          Defaults to None.
        eval_train (DataLoader, optional): DataLoader for evaluating training
          performance. Defaults to None.
    """
    print("--- Unlearning with NPO ---")
    # opt_save = copy.deepcopy(self.optimizer)
    self.og_model = copy.deepcopy(self.model)  # NEW
    repair_iters = self.opt.train_iters
    self.opt.train_iters = len(train_loader) + len(forget_loader)  # for Potion
    time_start = time.process_time()
    clean_retain_loader = train_loader
    self.curr_epoch = 50
    print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
    self.model.dropout.drop_targets = delete_idx  # needed for train eval
    prev_mode = self.model.dropout.drop_mode
    for eval_mode in ["drop"]:
      self.model.dropout.drop_mode = eval_mode
      print(
          "UL SPECIAL drop_mode set back to inference mode (test): ",
          self.model.dropout.drop_mode,
      )
      acc_adv_train = self.eval(
          forget_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=True,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/forget_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      acc_adv_train = self.eval(
          clean_retain_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=False,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/retain_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      acc_adv_train = self.eval(
          test_loader,
          save_model=False,
          train_fix=False,
          class_0=False,
          ic=False,
          only_infgti=False,
      )
      self.writer.add_scalar(
          f"UL_{eval_mode}/test_{eval_mode}", acc_adv_train, self.curr_epoch
      )
      if eval_train is not None:
        acc_adv_train = self.eval(
            eval_train,
            save_model=False,
            train_fix=False,
            class_0=False,
            ic=False,
            only_infgti=True,
        )
        self.writer.add_scalar(
            f"UL_{eval_mode}/healed_{eval_mode}", acc_adv_train, self.curr_epoch
        )
    self.curr_epoch += 1
    self.opt.train_iters = repair_iters  # flipping it back after potion
    # self.optimizer = opt_save
    self.optimizer = torch.optim.SGD(
        self.model.parameters(),
        lr=self.opt.unlearn_lr,  # self.opt.max_lr, #
        momentum=0.9,
        weight_decay=self.opt.wd,
    )
    self.scheduler = torch.optim.lr_scheduler.LinearLR(
        self.optimizer,
        start_factor=config.start_factor,
        end_factor=config.end_factor,
        total_iters=int(self.opt.train_iters / 97),
    )
    # self.model.dropout.drop_targets = delete_idx # discovered idxs
    print(
        "STEPS BEFORE START (curr/max):", self.curr_step, self.opt.train_iters
    )
    # warmup_counter = 0
    prev_proxy = 0
    loop_i = 0
    breaker = 0
    while self.curr_step < self.opt.train_iters:
      print("==> ", self.model.dropout.p_fixed, self.model.dropout.p_mem)
      time_start = time.process_time()
      self.model.dropout.drop_mode = "drop"
      print("NPO with drop mode: ", self.model.dropout.drop_mode)
      purge_acc = self.eval(
          forget_loader,
          save_model=False,
          train_fix=True,
          class_0=False,
          ic=False,
          only_infgti=True,
      )
      if loop_i == 0:
        self.best_model = copy.deepcopy(self.model)
        if purge_acc <= config.rem_mod_channel_threshold_purge:
          print("### Breaker loop activated as acc below desired already")
          # breaker = 1
          purge_acc = 100  # to ensure at least one step.
          # The best model save ensures this does no harm if it is worse
        loop_i += 1
      if purge_acc > config.rem_mod_channel_threshold_purge:
        _, _ = self.train_one_epoch(
            loader=forget_loader, ascent=True, no_scheduler_step=True
        )
        for eval_mode in ["drop"]:
          self.model.dropout.drop_mode = eval_mode
          print(
              "UL SPECIAL drop_mode set back to inference mode (test): ",
              self.model.dropout.drop_mode,
          )
          acc_1 = self.eval(
              forget_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/forget_{eval_mode}", acc_1, self.curr_epoch
          )
          acc_2 = self.eval(
              clean_retain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/retain_{eval_mode}", acc_2, self.curr_epoch
          )
          acc_3 = self.eval(
              pretrain_loader,
              save_model=False,
              train_fix=True,
              class_0=False,
              ic=False,
              only_infgti=True,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/pretrain_manip_overfit_{eval_mode}",
              acc_3,
              self.curr_epoch,
          )
          acc_4 = self.eval(
              test_loader,
              save_model=False,
              train_fix=False,
              class_0=False,
              ic=False,
              only_infgti=False,
          )
          self.writer.add_scalar(
              f"UL_{eval_mode}/test_{eval_mode}", acc_4, self.curr_epoch
          )
          if eval_train is not None:
            acc_5 = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}", acc_5, self.curr_epoch
            )
      else:
        print("Min acc reached")
        for eval_mode in ["drop"]:
          self.model.dropout.drop_mode = eval_mode
          if eval_train is not None:
            acc_5 = self.eval(
                eval_train,
                save_model=False,
                train_fix=False,
                class_0=False,
                ic=False,
                only_infgti=False,
            )
            self.writer.add_scalar(
                f"UL_{eval_mode}/healed_{eval_mode}", acc_5, self.curr_epoch
            )
        break
      # back to eval for final logging
      self.model.dropout.drop_mode = (
          prev_mode  # setting back for eval/inference
      )
      if breaker == 1:
        break
      # =========================
      # test if better than before but only using what we are given
      keeping = self.eval(clean_retain_loader, save_model=False, train_fix=True)
      removing = self.eval(forget_loader, save_model=False, train_fix=True)
      if removing < config.removing_threshold:
        removing = config.removing_threshold
      proxy = keeping * (1 - removing)
      self.writer.add_scalar("intermediate/proxy", proxy, self.curr_epoch)
      if proxy > prev_proxy:
        self.best_model = copy.deepcopy(self.model)
        prev_proxy = proxy
      # ========================
      self.curr_epoch += 1
      self.curr_step += 98  # since we do not have full ones with forget set
    # --------------
    self.save_files["train_time_taken"][0] += time.process_time() - time_start
    # train_loader = clean_retain_loader
    self.model = self.best_model
    return

  def get_save_prefix(self):
    self.unlearn_file_prefix = (
        self.opt.pretrain_file_prefix
        + "/"
        + str(self.opt.deletion_size)
        + "_"
        + self.opt.unlearn_method
        + "_"
        + self.opt.exp_name
    )
    self.unlearn_file_prefix += (
        "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
    )
    self.unlearn_file_prefix += config.run_name
    return
