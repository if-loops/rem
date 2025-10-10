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

"""Utility functions and classes for the unlearning framework.

Adapted from "Corrective Unlearning" and "Potion:Towards Poison Unlearning"
GitHub repos. https://github.com/drimpossible/corrective-unlearning-bench
https://github.com/if-loops/towards_poison_unlearning
"""

import copy
import functools
import os
import pickle
import random
from typing import Dict
import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.data import sampler
import tqdm

Sampler = sampler.Sampler
DataLoader = data.DataLoader
_LRScheduler = lr_scheduler.LRScheduler
partial = functools.partial
PYTORCH_MPS_HIGH_WATERMARK_RATIO = 0.0
device = "cuda" if torch.cuda.is_available() else "mps"


# Reference: https://github.com/if-loops/selective-synaptic-dampening/
# blob/main/src/forget_random_strategies.py
class ParameterPerturberOld:
  """Perturbs model parameters based on calculated importances.

  This class implements a Hessian-based method, more efficient than Fisher, to
  perturb model parameters. It's adapted from the selective synaptic dampening
  work. (Note: Old as it is the version used in the original SSD paper and not
  follow on works such as LFSSD or Potion)
  """

  def __init__(
      self,
      model,
      opt,
      device_,
      parameters=None,
  ):
    self.model = model
    self.opt = opt
    self.device = device_
    self.alpha = None
    self.xmin = None
    print(parameters)
    self.lower_bound = parameters["lower_bound"]
    self.exponent = parameters["exponent"]
    self.magnitude_diff = parameters["magnitude_diff"]  # unused
    self.min_layer = parameters["min_layer"]
    self.max_layer = parameters["max_layer"]
    self.forget_threshold = parameters["forget_threshold"]  # unused
    self.dampening_constant = parameters["dampening_constant"]  # lambda
    self.selection_weighting = parameters["selection_weighting"]  # alpha

  def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
    """Taken from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Returns a dict like
    named_parameters(), with zeroed-out parameter valuse

    Args:
      model (torch.nn): model to get param dict from

    Returns:
      dict(str,torch.Tensor): dict of zero-like params
    """
    return dict([
        (k, torch.zeros_like(p, device=p.device))
        for k, p in model.named_parameters()
    ])

  def calc_importance(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
    """Adapated from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Calculate per-parameter, importance

        returns a dictionary [param_name: list(importance per parameter)]
    Args:
      dataloader (DataLoader): DataLoader to be iterated over

    Returns:
      importances (dict(str, torch.Tensor([]))): named_parameters
      like dictionary containing list of importances for each param
    """
    criterion = nn.CrossEntropyLoss()
    importances = self.zerolike_params_dict(self.model)
    for x, y, idx in tqdm.tqdm(dataloader):
      x, y = x.to(self.device), y.to(self.device)
      self.opt.zero_grad()
      out = self.model(x, idx)
      loss = criterion(out, y)
      # override with ALFSSD loss
      # loss = torch.norm(out, p="fro", dim=1).abs().mean()
      loss.backward()
      for (_, p), (_, imp) in zip(
          self.model.named_parameters(), importances.items()
      ):
        if p.grad is not None:
          imp.data += p.grad.data.clone().pow(2)  # original
          # imp.data += p.grad.data.clone().abs()
    # average over mini batch length
    for _, imp in importances.items():
      imp.data /= float(len(dataloader))
    return importances

  def modify_weight(
      self,
      original_importance: Dict[str, torch.Tensor],
      forget_importance: Dict[str, torch.Tensor],
  ) -> None:
    """Perturb weights based on the SSD equations given in the paper.

    Args:
      original_importance (Dict[str, torch.Tensor]): dictionary of imps for
        original dataset
      forget_importance (Dict[str, torch.Tensor]): dictionary of importances for
        forget sample

    Returns:
      None
    """
    with torch.no_grad():
      for (_, p), (_, oimp), (_, fimp) in zip(
          self.model.named_parameters(),
          original_importance.items(),
          forget_importance.items(),
      ):
        # Synapse Selection with parameter alpha
        oimp_norm = oimp.mul(self.selection_weighting)
        locations = torch.where(fimp > oimp_norm)
        # Synapse Dampening with parameter lambda
        weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
            self.exponent
        )
        update = weight[locations]
        # Bound by 1 to prevent parameter values to increase.
        min_locs = torch.where(update > self.lower_bound)
        update[min_locs] = self.lower_bound
        p[locations] = p[locations].mul(update)


def convert_boolean_to_int(df):
  for column in df.columns:
    if df[column].dtype == bool:
      df[column] = df[column].astype(int)
  return df


class ParameterPerturber:
  """Perturbs model parameters based on calculated importances.

  This class implements methods for calculating parameter importances from
  datasets and modifying model weights based on these importances, following
  principles from SSD (Selective Synaptic Dampening) and its adaptive and
  label-free variants.
  """

  def __init__(
      self,
      model,
      opt,
      device_="cuda" if torch.cuda.is_available() else "mps",
      parameters=None,
      adaptive=False,
      label_free=False,
      x_d=False,
  ):
    self.model = model.to(device_)
    self.opt = opt
    self.alpha = parameters["selection_weighting"]
    self.xmin = None
    self.device = device_
    self.adaptive = adaptive
    self.label_free = label_free
    self.x_d = x_d
    self.lower_bound = 1
    self.exponent = 1
    self.magnitude_diff = None  # unused
    self.min_layer = -1
    self.max_layer = -1
    self.forget_threshold = 1  # unused
    self.dampening_constant = "Adaptive"  # parameters["dampening_constant"]
    self.selection_weighting = "Adaptive"  # parameters["selection_weighting"]

  def get_layer_num(self, layer_name: str) -> int:
    layer_id = layer_name.split(".")[1]
    if layer_id.isnumeric():
      return int(layer_id)
    else:
      return -1

  def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:
    """Taken from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Returns a dict like
    named_parameters(), with zeroed-out parameter valuse

    Args:
      model (torch.nn): model to get param dict from

    Returns:
      dict(str,torch.Tensor): dict of zero-like params
    """
    return dict([
        (k, torch.zeros_like(p, device=p.device))
        for k, p in model.named_parameters()
    ])

  def calc_importance(
      self, dataloader: DataLoader, extra_noise=False
  ) -> Dict[str, torch.Tensor]:
    """Adapated from: Avalanche: an End-to-End Library for Continual Learning.

    https://github.com/ContinualAI/avalanche Calculate per-parameter, importance

        returns a dictionary [param_name: list(importance per parameter)]
    Args:
      dataloader (DataLoader): DataLoader to be iterated over.
      extra_noise (bool): Whether to add extra noise to the input.

    Returns:
      importances (dict(str, torch.Tensor([]))): named_params-like dict
        containing list of importances for each parameter.
    """
    criterion = nn.CrossEntropyLoss()
    importances = self.zerolike_params_dict(self.model)
    for x, y, _, idx in dataloader:
      x, y = x.to(self.device), y.to(self.device)
      if extra_noise:
        print("NOISE IS ON")
        # add torch rand x% noise
        x += torch.randn_like(x) * 0.01
      self.opt.zero_grad()
      out = self.model(x, idx)
      if self.label_free:
        if self.x_d:
          # loss = torch.abs(out).sum(dim=1).mean() # L1
          # loss = torch.norm(out, p="fro", dim=1).abs().mean() # prev
          loss = torch.norm(out, p="fro", dim=1).abs().mean()
        else:
          # loss = torch.norm(out, p="fro", dim=1).abs().mean()
          loss = torch.norm(out, p="fro", dim=1).pow(2).mean()  # actual one
          # loss = (torch.norm(out, p="fro", dim=1).pow(2).mean())
          # # original SSD is pow(2) not abs -> l2 vs l1 norm
      else:
        loss = criterion(out, y)
      loss.backward()
      for (_, p), (_, imp) in zip(
          self.model.named_parameters(), importances.items()
      ):
        if p.grad is not None:
          if self.label_free:
            if self.x_d:
              # print("Using x_d")
              imp.data += p.grad.data.clone().abs()
              # imp.data += (p.grad.data.clone().abs()) # prev
            else:  # original is abs
              # print("Using original LF")
              imp.data += p.grad.data.clone().abs()
          else:
            # print("Using original SSD")
            imp.data += p.grad.data.clone().pow(2)
    # average over mini batch length
    for _, imp in importances.items():
      imp.data /= float(len(dataloader))
    return importances

  def modify_weight(
      self,
      original_importance: Dict[str, torch.Tensor],
      forget_importance: Dict[str, torch.Tensor],
      percentile_val="PERCENTILE NOT AUTOMATICALLY SET BUT ADAPTIVE SELECTED - ERROR",
      x_d=False,
  ) -> None:
    """Perturb weights based on the SSD equations given in the paper.

    Args:
      original_importance (Dict[str, torch.Tensor]): dict of importances for
        original dataset.
      forget_importance (Dict[str, torch.Tensor]): dictionary of importances for
        forget sample.
      percentile_val (str): The percentile value used for adaptive selection.
      x_d (bool): Whether to use the x_d variant of the loss.

    Returns:
      None
    """
    self.x_d = x_d
    if self.adaptive:  # adaptive SSD
      rel_list = list()
      # Get the indices of the fully connected layers
      fully_connected_layer_indices = list()
      for idx, layer in enumerate(self.model.children()):
        if isinstance(layer, nn.Linear):
          fully_connected_layer_indices.append(idx + 1)
      all_relative_values = []
      with torch.no_grad():
        for (_, p), (_, oimp), (_, fimp) in zip(
            self.model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
          layer_size_cutoff = 0  # overrride to do all layers
          if p.shape[0] >= layer_size_cutoff:  # only look at large layers
            divs_ = fimp.div(oimp)
            # select only the non nan values of divs_ to avoid errors
            divs_ = divs_[~torch.isnan(divs_)]
            # remove inf
            divs_ = divs_[~torch.isinf(divs_)]
            all_relative_values.append(divs_.reshape(-1).cpu().numpy())
      all_relative_values = np.concatenate(
          all_relative_values
      )  # flatten the array
      # PERCENTILE = NONE
      # print("USED Percentile: ", percentile_val)
      print("percentile:", percentile_val)
      print(all_relative_values)
      percentile = np.nanpercentile(all_relative_values, percentile_val)
      # print("USED cutoff value: ", percentile)
      percentile = percentile.item()
      print("percentile value:", percentile)
      # Main part after finding cutoff value
      with torch.no_grad():
        for (_, p), (_, oimp), (_, fimp) in zip(
            self.model.named_parameters(),
            original_importance.items(),
            forget_importance.items(),
        ):
          # divs_ = fimp.div(oimp)
          # select only the non nan values of divs_
          # divs_ = divs_[~torch.isnan(divs_)]
          # print(fimp, "XXX", oimp)
          # relative = torch.mean(divs_)
          # rel_std = torch.std(divs_)
          # rel_median = torch.median(divs_)
          # calculate absolute difference between median and mean
          # abs_diff = torch.abs(rel_median - relative)
          # Always adaptive
          self.selection_weighting = percentile
          self.dampening_constant = 1  # constant from ASSD paper
          rel_list.append(self.selection_weighting)
          # Synapse Selection with parameter alpha
          oimp_norm = oimp.mul(self.selection_weighting)
          locations = torch.where(fimp > oimp_norm)
          # Synapse Dampening with parameter lambda
          weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(
              self.exponent
          )
          update = weight[locations]
          # Bound by 1 to prevent parameter values to increase.
          min_locs = torch.where(update > self.lower_bound)
          # for update take the update value where update > 0.1,
          # otherwise set update 0.1
          # We do not use this in the paper but this can be used
          # to avoid dead nerons for extra robustness
          # if x_d:
          #    dampen_limit = 0.01
          # else:
          #    dampen_limit = 0
          dampen_limit = 0
          update[update < dampen_limit] = dampen_limit
          update[min_locs] = self.lower_bound
          p[locations] = p[locations].mul(update)


def ssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device_,
):
  """Performs SSD (Selective Synaptic Dampening) tuning on the model.

  This function calculates importances for the forget set and the full training
  set, and then modifies the model weights based on these importances using the
  SSD method.

  Args:
    model: The PyTorch model to be tuned.
    forget_train_dl: DataLoader for the forget set.
    dampening_constant: The dampening constant (lambda) for SSD.
    selection_weighting: The selection weighting (alpha) for SSD.
    full_train_dl: DataLoader for the full training set.
    device_: The device to run the computations on ('cuda' or 'mps').

  Returns:
    The modified PyTorch model.
  """
  parameters = {
      "lower_bound": 1,
      "exponent": 1,
      "magnitude_diff": None,
      "min_layer": -1,
      "max_layer": -1,
      "forget_threshold": 1,
      "dampening_constant": dampening_constant,
      "selection_weighting": selection_weighting,
  }
  # load the trained model
  optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
  pdr = ParameterPerturberOld(model, optimizer, device_, parameters)
  model = model.eval()
  sample_importances = pdr.calc_importance(forget_train_dl)
  original_importances = pdr.calc_importance(full_train_dl)
  pdr.modify_weight(original_importances, sample_importances)
  return model


def assd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device_,
    frac_dl,
):
  """Performs Adaptive SSD (ASSD) tuning on the model.

  This function calculates importances for the forget set and a fraction of the
  full training set, and then adaptively modifies the model weights based on
  these importances using the ASSD method. It includes an option to load
  pre-calculated importances for speedup.

  Args:
    model: The PyTorch model to be tuned.
    forget_train_dl: DataLoader for the forget set.
    dampening_constant: Placeholder for dampening constant (adaptive).
    selection_weighting: Placeholder for selection weighting (adaptive).
    full_train_dl: DataLoader for the full training set.
    device_: The device to run the computations on ('cuda' or 'mps').
    frac_dl: The frac of the dataset used for adaptive percentile calc.

  Returns:
    The modified PyTorch model.
  """
  _ = dampening_constant  # not used in adaptive
  _ = selection_weighting
  parameters = {
      "lower_bound": 1,
      "exponent": 1,
      "magnitude_diff": None,
      "min_layer": -1,
      "max_layer": -1,
      "forget_threshold": 1,
      "dampening_constant": None,  # adaptive overwrites this
      "selection_weighting": None,  # adaptive overwrites this
  }
  print("----- Using ASSD -----")
  # Sweep loop (ASSD extension)
  sweeps_n = 1
  frac_dl = frac_dl * (1 / sweeps_n)
  for _ in range(sweeps_n):
    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ParameterPerturber(
        model, optimizer, device_, parameters, adaptive=True
    )
    model = model.eval()
    ########################## SPEEDUP
    file_name = "original_importances.pkl"
    if os.path.exists(file_name):
      print("##### LOADING IMPORTANCES")
      with open("original_importances.pkl", "rb") as f:
        original_importances = pickle.load(f)
      with open("sample_importances.pkl", "rb") as f:
        sample_importances = pickle.load(f)
    else:
      # ----
      print("##### CALCULATING IMPORTANCES")
      # Calculate the importances of D (see paper)
      original_importances = pdr.calc_importance(full_train_dl)
      # safe the importances locally for reuse
      with open("original_importances.pkl", "wb") as f:
        pickle.dump(original_importances, f)
      # Calculation of the forget set importances
      sample_importances = pdr.calc_importance(forget_train_dl)
      with open("sample_importances.pkl", "wb") as f:
        pickle.dump(sample_importances, f)
    ########################## SPEEDUP
    # auto select percentile
    # len_all = len(full_train_dl.dataset) + len(forget_train_dl.dataset)
    # len_forget = len(forget_train_dl.dataset)
    # len_forget = len(forget_train_dl.dataset)
    # len_all = len(full_train_dl.dataset) + len(forget_train_dl.dataset)
    share_off = np.log(1 + frac_dl * 100)
    percentile = 100 - share_off
    print("###### ----- Length based percentile: ", percentile)
    # Dampen selected parameters
    _ = pdr.modify_weight(
        original_importances, sample_importances, percentile_val=percentile
    )
  return model


# placeholder for now reusing assd
def alfssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device_,
    frac_dl,
    filtered_loader,
    x_d=False,
    optimizer=None,
    original_importances=None,
    sample_importances=None,
):
  """Performs Adaptive Label-Free SSD (ALFSSD) tuning on the model.

  This function calculates importances for a filtered dataset and the forget set
  using a label-free approach, and then adaptively modifies the model weights
  based on these importances. It supports an option for an 'x_d' variant.
  Pre-calculated importances can be provided to speed up the process.

  Args:
    model: The PyTorch model to be tuned.
    forget_train_dl: DataLoader for the forget set.
    dampening_constant: Unused in ALFSSD, as it's adaptive.
    selection_weighting: Unused in ALFSSD, as it's adaptive.
    full_train_dl: Unused in this function.
    device_: The device to run the computations on ('cuda' or 'mps').
    frac_dl: The fraction of the dataset used for adaptive percentile calc.
    filtered_loader: DataLoader for the filtered dataset used to calculate
      original importances.
    x_d (bool): Whether to use the x_d variant of the loss.
    optimizer: Optional optimizer. If None, a new SGD optimizer is created.
    original_importances: Optional pre-calculated importances for the original
      dataset.
    sample_importances: Optional pre-calculated importances for the forget set.

  Returns:
    A tuple containing:
      - model: The modified PyTorch model.
      - original_importances: The calculated or provided original importances.
      - sample_importances: The calculated or provided sample importances.
  """
  parameters = {
      "lower_bound": 1,
      "exponent": 1,
      "magnitude_diff": None,
      "min_layer": -1,
      "max_layer": -1,
      "forget_threshold": 1,
      "dampening_constant": None,  # adaptive
      "selection_weighting": None,
  }
  _ = (dampening_constant,)  # unused in alfssd
  _ = selection_weighting
  _ = full_train_dl
  # Sweep loop (ASSD extension)
  if x_d:
    sweeps_n = 1
    frac_dl = frac_dl * (1 / sweeps_n)
  else:
    sweeps_n = 1  # i.e. original
    frac_dl = frac_dl * (1 / sweeps_n)
  for sweep_i in range(sweeps_n):
    sweep_i += 1
    print("Sweep #", sweep_i)
    # load the trained model
    if optimizer is None:
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ParameterPerturber(
        model,
        optimizer,
        device_,
        parameters,
        adaptive=True,
        label_free=True,
        x_d=x_d,
    )
    model = model.eval()
    if original_importances:
      pass
    else:
      # ----
      # Calculate the importances of D (see paper)
      original_importances = pdr.calc_importance(
          filtered_loader, extra_noise=False
      )
      # safe the importances locally for reuse
      # with open("original_importances.pkl", "wb") as f:
      #    pickle.dump(original_importances, f)
      # Calculation of the forget set importances
      sample_importances = pdr.calc_importance(
          forget_train_dl, extra_noise=False
      )
    if x_d:
      share_off = np.log(1 + frac_dl * 100)  # orig 100
      percentile = 100 - share_off
    else:  # original
      share_off = np.log(1 + frac_dl * 100)
      percentile = 100 - share_off
    print("###### ----- Length based percentile: ", percentile)
    # Dampen selected parameters
    _ = pdr.modify_weight(
        original_importances,
        sample_importances,
        percentile_val=percentile,
        x_d=x_d,
    )
  return model, original_importances, sample_importances


class LinearLR(_LRScheduler):
  r"""Sets the learning rate of each parameter group with a linear schedule.

  The schedule follows :math:`\eta_{t} = \eta_0*(1 - t/T)`, where :math:`\eta_0`
  is the initial lr, :math:`t` is the current epoch or iteration (zero-based)
  and :math:`T` is the total training epochs or iterations. It is recommended to
  use the iteration based calculation if the total number of epochs is small.
  When last_epoch=-1, sets initial lr as lr. This scheduler is studied in:
  `Budgeted Training: Rethinking Deep Neural Network Training Under Resource

   Constraints`_.
  .. _Budgeted Training\: Rethinking Deep Neural Network Training Under
  Resource Constraints:
      https://arxiv.org/abs/1905.04753
  """

  def __init__(self, optimizer, t, warmup_epochs=100, last_epoch=-1):
    """Initializes the LinearLR scheduler.

    Args:
      optimizer (Optimizer): Wrapped optimizer.
      t (int): Total number of training epochs or iterations.
      warmup_epochs (int): Number of epochs for the linear warmup phase.
      last_epoch (int): The index of last epoch or iteration. Default: -1.
    """
    self.t = float(t)
    self.warm_ep = warmup_epochs
    super(LinearLR, self).__init__(optimizer, last_epoch)

  def get_lr(self):
    if self.last_epoch - self.warm_ep >= 0:
      rate = 1 - ((self.last_epoch - self.warm_ep) / self.t)
    else:
      rate = (self.last_epoch + 1) / (self.warm_ep + 1)
    return [rate * base_lr for base_lr in self.base_lrs]

  def _get_closed_form_lr(self):
    return self.get_lr()


def cutmix(x, y, alpha=1.0):
  """Applies the CutMix augmentation to a batch of images and labels.

  CutMix involves cutting and pasting patches between training images. The
  labels are also mixed proportionally to the area of the patches.

  Args:
    x: Input tensor of images.
    y: Input tensor of labels.
    alpha: The alpha parameter for the Beta distribution, controlling the size
      of the cutouts.

  Returns:
    A tuple containing:
      - x: The mixed image tensor.
      - y_a: The original labels.
      - y_b: The labels from the mixed-in image.
      - lam: The mixing ratio.
  """
  assert alpha > 0
  # generate mixed sample
  lam = np.random.beta(alpha, alpha)
  batch_size = x.size()[0]
  index = torch.randperm(batch_size).to(device)  # .cuda()
  y_a, y_b = y, y[index]
  bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
  x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
  # adjust lambda to exactly match pixel ratio
  lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
  return x, y_a, y_b, lam


def rand_bbox(size, lam):
  """Generates a random bounding box for CutMix.

  Args:
    size (tuple): The size of the input tensor.
    lam (float): The mixing ratio from the Beta distribution.

  Returns:
    tuple: Coordinates of the bounding box (bbx1, bby1, bbx2, bby2).
  """
  w = size[2]
  h = size[3]
  cut_rat = np.sqrt(1.0 - lam)
  cut_w = np.int32(w * cut_rat)
  cut_h = np.int32(h * cut_rat)
  # uniform
  cx = np.random.randint(w)
  cy = np.random.randint(h)
  bbx1 = np.clip(cx - cut_w // 2, 0, w)
  bby1 = np.clip(cy - cut_h // 2, 0, h)
  bbx2 = np.clip(cx + cut_w // 2, 0, w)
  bby2 = np.clip(cy + cut_h // 2, 0, h)
  return bbx1, bby1, bbx2, bby2


def seed_everything(seed):
  """Function that sets seed for pseudo-random number generators.

  Args:
      seed: the integer value seed for global random state

  Returns:
      The integer value seed.
  """
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  return seed


class ModelWithTTA(torch.nn.Module):
  """A wrapper class for a model to perform Test-Time Augmentation (TTA).

  This class takes a base model and applies TTA, currently supporting horizontal
  flipping, to the input. The outputs from the original and augmented inputs are
  combined.
  """

  def __init__(self, model, mult_fact=1, tta_type="flip"):
    super().__init__()
    self.model = model
    self.mult_fact = mult_fact
    self.tta_type = tta_type

  def forward(self, x):
    out = self.model(x, torch.zeros(x.size(0), device=device)) * self.mult_fact
    if self.tta_type == "flip":
      out += self.model(
          torch.flip(x, dims=[3]), torch.zeros(x.size(0), device=device)
      )
      out /= 2
    return self.net(out)


def get_targeted_classes(dataset):
  """Returns a tuple of targeted class indices for given datasets.

  Args:
    dataset (str): The name of the dataset.

  Returns:
    tuple: A tuple containing the indices of the targeted classes.
  Raises:
    ValueError: If the dataset is not recognized.
    AssertionError: If the dataset is "LFWPeople" or "CelebA", as these are
      not yet implemented.
  """
  if dataset == "CIFAR10":
    classes = (3, 5)
  elif dataset == "tinyimagenet":
    classes = (3, 5)
  elif dataset == "Imagenette":
    classes = (3, 5)
  elif dataset == "CIFAR100":
    classes = (47, 53)
  elif dataset == "SVHN":
    classes = (3, 5)
  elif dataset in ["PCAM", "DermNet", "Pneumonia"]:
    classes = (0, 1)
  elif dataset in ["LFWPeople", "CelebA"]:
    # Raise NotImplemented Error
    assert False, "Not Implemented Yet"
  else:
    raise ValueError(f"Dataset '{dataset}' not recognized.")
  return classes


def unlearn_func(
    model,
    method,
    factor=0.1,
    device_="cuda" if torch.cuda.is_available() else "mps",
):
  """Applies unlearning to the model based on the specified method.

  This function creates a copy of the model and modifies its weights based on
  the chosen unlearning method ("EU" or "Mixed").

  Args:
    model: The PyTorch model to be unlearned.
    method (str): The unlearning method to apply. Can be "EU" (Erasure
      Unlearning) or "Mixed".
    factor (float): The scaling factor used in the "Mixed" method.
    device_ (str): The device to move the model to after modification. Defaults
      to "cuda" if available, otherwise "mps".

  Returns:
    torch.nn.Module: The modified PyTorch model.
  """
  model = copy.deepcopy(model)
  model = model.cpu()
  if method == "EU":
    model.apply(initialize_weights)
  elif method == "Mixed":
    partialfunc = partial(modify_weights, factor=factor)
    model.apply(partialfunc)
  else:
    pass
  model.to(device_)
  return model


def initialize_weights(m):
  if isinstance(m, torch.nn.Conv2d):
    m.reset_parameters()
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)
  elif isinstance(m, torch.nn.BatchNorm2d):
    m.reset_parameters()
  elif isinstance(m, torch.nn.Linear):
    m.reset_parameters()
    if m.bias is not None:
      torch.nn.init.constant_(m.bias.data, 0)


def modify_weights(m, factor=0.1):
  """Modifies the weights and biases of a given module by a scaling factor.

  This function is used within the "Mixed" unlearning method to scale down the
  parameters of Conv2d, BatchNorm2d, and Linear layers.

  Args:
    m: The torch.nn.Module to modify.
    factor (float): The scaling factor to apply to the weights and biases.
  """
  if isinstance(m, torch.nn.Conv2d):
    m.weight.data = m.weight.data * factor
    if m.bias is not None:
      m.bias.data = m.bias.data * factor
  elif isinstance(m, torch.nn.BatchNorm2d):
    if m.affine:
      m.weight.data = m.weight.data * factor
      m.bias.data = m.bias.data * factor
  elif isinstance(m, torch.nn.Linear):
    m.weight.data = m.weight.data * factor
    if m.bias is not None:
      m.bias.data = m.bias.data * factor


def distill_kl_loss(y_s, y_t, t, reduction="sum"):
  p_s = torch.nn.functional.log_softmax(y_s / t, dim=1)
  p_t = torch.nn.functional.softmax(y_t / t, dim=1)
  loss = torch.nn.functional.kl_div(p_s, p_t, reduction=reduction)
  if reduction == "none":
    loss = torch.sum(loss, dim=1)
  loss = loss * (t**2) / y_s.shape[0]
  return loss


def compute_accuracy(preds, y):
  return np.equal(np.argmax(preds, axis=1), y).mean()


class SubsetSequentialSampler(Sampler):

  def __init__(self, indices):
    self.indices = indices

  def __iter__(self):
    return (self.indices[i] for i in range(len(self.indices)))

  def __len__(self):
    return len(self.indices)
