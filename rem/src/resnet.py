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

"""A collection of ResNet and Vision Transformer (ViT) models.

The ResNet implementations are based on the repository (hence the name):
https://github.com/drimpossible/corrective-unlearning-bench
"""

import re
import config
import einops
import einops.layers.torch
from methods import ExampleTiedDropout
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import initialize_weights

repeat = einops.repeat
rearrange = einops.rearrange
Rearrange = einops.layers.torch.Rearrange
__all__ = [
    "ResNet",
    "resnet9",
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "resnet110",
    "resnetwide28x10",
]
track_running_stats = config.bn_track_running_stats
momentum = config.bn_momentum


class BasicBlock(nn.Module):
  """Basic Block for ResNet."""

  expansion = 1

  def __init__(self, in_planes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = nn.BatchNorm2d(
        in_planes, track_running_stats=track_running_stats, momentum=momentum
    )
    self.conv2 = nn.Conv2d(
        planes, planes, kernel_size=3, stride=1, padding=1, bias=False
    )
    self.bn2 = nn.BatchNorm2d(
        planes, track_running_stats=track_running_stats, momentum=momentum
    )
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(
              in_planes,
              self.expansion * planes,
              kernel_size=1,
              stride=stride,
              bias=False,
          ),
          nn.BatchNorm2d(
              self.expansion * planes,
              track_running_stats=track_running_stats,
              momentum=momentum,
          ),
      )

  def forward(self, x):
    out = F.relu(self.conv1(self.bn1(x)))
    out = self.conv2(self.bn2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class ResNet(nn.Module):
  """A ResNet model.

  Attributes:
      in_planes: The number of input planes for the current layer.
      conv1: The initial convolutional block.
      layer1: The first set of residual blocks.
      layer2: The second set of residual blocks.
      layer3: The third set of residual blocks.
      norm: The average pooling and flattening layers.
      fc: The final fully connected layer for classification.
  """

  def __init__(self, block, num_blocks, num_classes=10):
    """Initializes the ResNet model.

    Args:
        block: The building block for the ResNet (e.g., BasicBlock).
        num_blocks: A list specifying the number of blocks in each of the three
          main layers.
        num_classes: The number of output classes.
    """
    super(ResNet, self).__init__()
    self.in_planes = 16
    self.conv1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(16, track_running_stats=track_running_stats),
        nn.ReLU(inplace=True),
    )
    self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
    self.norm = self.norm = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
    self.fc = nn.Linear(64, num_classes)
    # self.dropout = ExampleTiedDropout()

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.conv1(x)
    out = self.layer1(out)
    # out = self.dropout(out, idx)
    out = self.layer2(out)
    # out = self.dropout(out, idx)
    out = self.layer3(out)
    out = self.norm(out)
    out = self.fc(out)
    return out


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
  """Creates a convolutional block with BatchNorm and CELU activation.

  Args:
      in_channels: Number of input channels.
      out_channels: Number of output channels.
      kernel_size: Size of the convolutional kernel.
      stride: Stride of the convolution.
      padding: Padding to be added to the input.

  Returns:
      A nn.Sequential containing Conv2d, BatchNorm2d, and CELU.
  """
  layers = [
      nn.Conv2d(
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
      ),
      nn.BatchNorm2d(
          out_channels,
          track_running_stats=track_running_stats,
          momentum=momentum,
      ),
      nn.CELU(alpha=0.075),
  ]
  return nn.Sequential(*layers)


class Residual(nn.Module):

  def __init__(self, module):
    super(Residual, self).__init__()
    self.module = module

  def forward(self, x):
    return x + self.module(x)


class ResNet9(nn.Module):
  """A ResNet-9 model architecture.

  This class implements a ResNet-9 model, including convolutional layers,
  residual blocks, and a final fully connected layer. It also incorporates
  ExampleTiedDropout (ETD) for regularization, configured via `etd_args`.
  """

  def __init__(self, num_classes: int = 10, etd_args: str = "None"):
    super(ResNet9, self).__init__()
    self.conv1 = conv_bn(3, 64)
    self.conv2 = conv_bn(64, 128, 5, 2, 2)
    self.res1 = Residual(
        nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )
    )
    self.conv3 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
    self.res2 = Residual(
        nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )
    )
    self.conv4 = nn.Sequential(
        conv_bn(256, 128, 3, 1, 0), nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten()
    )
    self.fc = nn.Linear(128, num_classes, bias=False)
    # dummy
    self.drop_mode = "Not used"
    print(f"==> Setting up vanilla model with drop mode: {self.drop_mode}")
    self.dropout = ExampleTiedDropout(p_fixed=1, p_mem=0)
    self.ttype_store = etd_args

  def forward(self, x, idx, epoch=None, classidx=None):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out)
    out = self.conv3(out)
    out = self.res2(out)
    out = self.conv4(out)
    out = self.fc(out)
    return out


class ResNet9ETD(nn.Module):
  """A ResNet-9 model with ExampleTiedDropout (ETD).

  This class extends the ResNet-9 architecture by incorporating
  ExampleTiedDropout layers within the forward pass, configured based on the
  provided `etd_args`.
  """

  def __init__(self, num_classes: int = 10, etd_args: str = "None"):
    super(ResNet9ETD, self).__init__()
    self.conv1 = conv_bn(3, 64)
    self.conv2 = conv_bn(64, 128, 5, 2, 2)
    self.res1 = Residual(
        nn.Sequential(
            conv_bn(128, 128),
            conv_bn(128, 128),
        )
    )
    self.conv3 = nn.Sequential(conv_bn(128, 256), nn.MaxPool2d(2))
    self.res2 = Residual(
        nn.Sequential(
            conv_bn(256, 256),
            conv_bn(256, 256),
        )
    )
    # Original
    # self.conv4 = nn.Sequential(conv_bn(256, 128, 3, 1, 0),
    # nn.AdaptiveMaxPool2d((1, 1)),
    # nn.Flatten())
    # Split to insert ETD
    self.conv4 = conv_bn(256, 128, 3, 1, 0)
    self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)), nn.Flatten())
    self.fc = nn.Linear(128, num_classes, bias=False)
    train_type_split = etd_args.split("_")
    relevance_part = [part for part in train_type_split if "ETD" in part][0]
    match_fixed = re.search(r"ETDfix(\d+)", relevance_part)
    match_mem = re.search(r"mem(\d+)", relevance_part)
    if match_fixed is None:
      raise ValueError(f"Could not find ETDfix in {relevance_part}")
    if match_mem is None:
      raise ValueError(f"Could not find mem in {relevance_part}")
    self.p_fixed = float(match_fixed.group(1)) / 100
    self.p_mem = float(match_mem.group(1)) / 100
    # ------------ ETD
    train_type_split = etd_args.split("_")
    try:
      relevance_part = [part for part in train_type_split if "ETD" in part][0]
    except IndexError:
      relevance_part = [part for part in train_type_split if "ETD" in part]
    match_drop = re.search(r"drop(\d+)", relevance_part)
    if match_drop is None:
      raise ValueError(f"Could not find drop in {relevance_part}")
    self.match_drop = str(match_drop.group(1))
    if self.match_drop == "0":  # pass all neurons through
      self.drop_mode = "test"
    elif self.match_drop == "1":  # drop all mem neurons
      self.drop_mode = "drop"
    elif self.match_drop == "2":  # extra option to be used
      self.drop_mode = "drop"
    elif self.match_drop == "3":  # extra option to be used
      self.drop_mode = "test"
    elif self.match_drop == "4":  # extra option to be used
      self.drop_mode = "targetdrop"
    elif self.match_drop == "5":  # extra option to be used
      self.drop_mode = "test"
    elif self.match_drop == "8":  # extra option to be used
      self.drop_mode = "drop"
    elif self.match_drop == "9":  # extra option to be used
      self.drop_mode = "train"
    else:
      self.drop_mode = "test"
    print("Setting up model with drop mode: ", self.drop_mode)
    # --------------------
    if hasattr(self, "dropout"):
      print(
          "ETD was already set up for this model. Reusing ETD params ",
          self.p_fixed,
          self.p_mem,
      )
      self.dropout.drop_mode = (
          self.drop_mode
      )  # just to be safe to not reuse the train args
    else:
      print("NO ETD AVAILABLE. Setting up new ETD ", self.p_fixed, self.p_mem)
      self.dropout = ExampleTiedDropout(
          p_fixed=self.p_fixed, p_mem=self.p_mem, drop_mode=self.drop_mode
      )
    self.ttype_store = etd_args
    self.dropout.rand_dropout_on = config.rand_dropout_on
    self.dropout.rand_dropout_p = config.rand_dropout_p
    if self.dropout.rand_dropout_on:
      self.og_dropout = torch.nn.Dropout(p=self.rand_dropout_p)

  def forward(self, x, idx, epoch=None, classidx=None):
    out = self.conv1(x)
    out = self.dropout(
        out, idx, epoch, classidx=classidx, pair=0
    )  # can't pair all of them as shapes change after conv
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[0][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.conv2(out)
    out = self.dropout(out, idx, epoch, classidx=classidx, pair=1)
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[1][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.res1(out)
    out = self.dropout(out, idx, epoch, classidx=classidx, pair=1)  # Pair A
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[1][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.conv3(out)
    out = self.dropout(out, idx, epoch, classidx=classidx, pair=2)  # Pair B
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[2][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.res2(out)
    out = self.dropout(out, idx, epoch, classidx=classidx, pair=2)  # Pair B
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[2][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.conv4(out)
    out = self.dropout(out, idx, epoch, classidx=classidx, pair=3)
    if self.dropout.rand_dropout_on:
      etd_mask = self.dropout.mask_tensor[3][idx]  # interger is the pair
      self.og_dropout(torch.ones_like(out))
      out = torch.where(~etd_mask, out, out * etd_mask)
    out = self.pool(out)
    out = self.fc(out)
    return out


class WideBasic(nn.Module):
  """A basic building block for WideResNet."""

  def __init__(self, in_planes, planes, stride=1):
    super(WideBasic, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size=3, padding=1, bias=True
    )
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(
        planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
    )
    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
          nn.BatchNorm2d(planes),
      )

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)
    return out


class WideResNet(nn.Module):
  """A WideResNet model architecture.

  This class implements a Wide Residual Network, allowing for increased width
  (widen_factor) in its layers.
  """

  def __init__(
      self, num_classes=10, depth=28, widen_factor=10, etd_args: str = "None"
  ):
    super(WideResNet, self).__init__()
    self.in_planes = 16
    assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
    n = (depth - 4) / 6
    k = widen_factor
    nstages = [16, 16 * k, 32 * k, 64 * k]
    self.conv1 = nn.Conv2d(
        3, nstages[0], kernel_size=3, stride=1, padding=1, bias=True
    )
    self.layer1 = self._wide_layer(WideBasic, nstages[1], n, stride=1)
    self.layer2 = self._wide_layer(WideBasic, nstages[2], n, stride=2)
    self.layer3 = self._wide_layer(WideBasic, nstages[3], n, stride=2)
    self.norm = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
    self.fc = nn.Linear(nstages[3], num_classes)
    # dummy
    self.dropout = ExampleTiedDropout(p_fixed=100, p_mem=10)
    self.ttype_store = etd_args

  def _wide_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x, idx, epoch=None, classidx=None):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.norm(out)
    out = self.fc(out)
    return out


class WideResNetETD(nn.Module):
  """A WideResNet model with ExampleTiedDropout (ETD).

  This class implements a Wide Residual Network and integrates
  ExampleTiedDropout for regularization, configured via `etd_args`.
  """

  def __init__(
      self, num_classes=10, depth=28, widen_factor=10, etd_args: str = "None"
  ):
    super(WideResNetETD, self).__init__()
    self.in_planes = 16
    assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
    n = (depth - 4) / 6
    k = widen_factor
    nstages = [16, 16 * k, 32 * k, 64 * k]
    self.conv1 = nn.Conv2d(
        3, nstages[0], kernel_size=3, stride=1, padding=1, bias=True
    )
    self.layer1 = self._wide_layer(WideBasic, nstages[1], n, stride=1)
    self.layer2 = self._wide_layer(WideBasic, nstages[2], n, stride=2)
    self.layer3 = self._wide_layer(WideBasic, nstages[3], n, stride=2)
    self.norm = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
    self.fc = nn.Linear(nstages[3], num_classes)
    train_type_split = etd_args.split("_")
    relevance_part = [part for part in train_type_split if "ETD" in part][0]
    match_fixed = re.search(r"ETDfix(\d+)", relevance_part)
    match_mem = re.search(r"mem(\d+)", relevance_part)
    if match_fixed is None:
      raise ValueError(f"Could not find ETDfix in {relevance_part}")
    if match_mem is None:
      raise ValueError(f"Could not find mem in {relevance_part}")
    p_fixed = float(match_fixed.group(1)) / 100
    p_mem = float(match_mem.group(1)) / 100
    self.dropout = ExampleTiedDropout(p_fixed=p_fixed, p_mem=p_mem)
    self.ttype_store = etd_args

  def _wide_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * int(num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes
    return nn.Sequential(*layers)

  def forward(self, x, idx, epoch=None, classidx=None):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.dropout(out, idx, epoch, classidx=classidx)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.norm(out)
    out = self.fc(out)
    return out


def resnet20(num_classes=10):
  net = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
  net.embed_size = 64
  net.apply(initialize_weights)
  return net


def resnet32(num_classes=10):
  net = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
  net.embed_size = 64
  net.apply(initialize_weights)
  return net


def resnet44(num_classes=10):
  net = ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)
  net.embed_size = 64
  net.apply(initialize_weights)
  return net


def resnet56(num_classes=10):
  net = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
  net.embed_size = 64
  net.apply(initialize_weights)
  return net


def resnet110(num_classes=10):
  net = ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)
  net.embed_size = 64
  net.apply(initialize_weights)
  return net


def resnet9(num_classes=10, train_type=None):
  net = ResNet9(num_classes=num_classes, etd_args=train_type)
  net.embed_size = 128
  net.apply(initialize_weights)
  return net


def resnet9_etd(num_classes=10, train_type=None):
  net = ResNet9ETD(num_classes=num_classes, etd_args=train_type)
  net.embed_size = 128
  net.apply(initialize_weights)
  return net


def resnetwide28x10_etd(num_classes=10, train_type=None):
  net = WideResNetETD(
      depth=28, widen_factor=10, num_classes=num_classes, etd_args=train_type
  )
  net.embed_size = 640
  net.apply(initialize_weights)
  return net


def resnetwide28x10(num_classes=10, train_type=None):
  net = WideResNet(
      depth=28, widen_factor=10, num_classes=num_classes, etd_args=train_type
  )
  net.embed_size = 640
  net.apply(initialize_weights)
  return net


def test(net):
  total_params = 0
  inp = torch.randn(size=(1, 3, 32, 32))
  out = net(inp)
  print(f"output: {out.size()}")
  for x in filter(lambda p: p.requires_grad, net.parameters()):
    total_params += np.prod(x.data.numpy().shape)
  print("Total number of params", total_params)
  print(
      "Total layers",
      len(
          list(
              filter(
                  lambda p: p.requires_grad and len(p.data.size()) > 1,
                  net.parameters(),
              )
          )
      ),
  )


# ------------------------- ViT -------------------------
def vit(num_classes=10, train_type=None):
  net = VitClass(num_classes=num_classes, etd_args=train_type)
  # net.embed_size = 128
  net.apply(initialize_weights)
  return net


def vit_etd(num_classes=10, train_type=None):
  net = VitClass(num_classes=num_classes, etd_args=train_type)
  # net.embed_size = 128
  net.apply(initialize_weights)
  return net


# helpers
def pair(t):
  return t if isinstance(t, tuple) else (t, t)


# classes
class FeedForward(nn.Module):
  """A FeedForward block for Transformer models.

  This block consists of a LayerNorm, two linear layers with a GELU activation
  in between, and dropout layers.
  """

  def __init__(self, dim, hidden_dim, vanilla_dropout=0.0):
    super().__init__()
    self.net = nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(vanilla_dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(vanilla_dropout),
    )

  def forward(self, x):
    return self.net(x)


class LSA(nn.Module):
  """Local Self-Attention module.

  This module performs self-attention with a local mask, including query, key,
  and value projections, scaling, and dropout.
  """

  def __init__(self, dim, heads=8, dim_head=64, vanilla_dropout=0.0):
    super().__init__()
    inner_dim = dim_head * heads
    self.heads = heads
    self.temperature = nn.Parameter(torch.log(torch.tensor(dim_head**-0.5)))
    self.norm = nn.LayerNorm(dim)
    self.attend = nn.Softmax(dim=-1)
    self.vanilla_dropout = nn.Dropout(vanilla_dropout)
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
    self.to_out = nn.Sequential(
        nn.Linear(inner_dim, dim), nn.Dropout(vanilla_dropout)
    )

  def forward(self, x):
    x = self.norm(x)
    qkv = self.to_qkv(x).chunk(3, dim=-1)
    q, k, v = map(
        lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
    )
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.temperature.exp()
    mask = torch.eye(dots.shape[-1], device=dots.device, dtype=torch.bool)
    mask_value = -torch.finfo(dots.dtype).max
    dots = dots.masked_fill(mask, mask_value)
    attn = self.attend(dots)
    attn = self.vanilla_dropout(attn)
    out = torch.matmul(attn, v)
    out = rearrange(out, "b h n d -> b n (h d)")
    return self.to_out(out)


class Transformer(nn.Module):
  """A Transformer model composed of multiple layers of Self-Attention and FeedForward.

  Attributes:
      layers: A ModuleList containing pairs of LSA and FeedForward modules.
      dropout: The dropout module to be used, potentially an ExampleTiedDropout.
  """

  def __init__(self, dim, depth, heads, dim_head, mlp_dim, vanilla_dropout=0.0):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(
          nn.ModuleList([
              LSA(
                  dim,
                  heads=heads,
                  dim_head=dim_head,
                  vanilla_dropout=vanilla_dropout,
              ),
              FeedForward(dim, mlp_dim, vanilla_dropout=vanilla_dropout),
          ])
      )

  def forward(
      self, x, idx=None, epoch=None, classidx=None, etd_args=None, dropout=None
  ):
    pair_idx = 0
    self.dropout = dropout
    for attn, ff in self.layers:
      # if etd_args != "Naive":
      #    x = self.dropout(x, idx, epoch, classidx=classidx, pair=pair_idx)
      # # use same pair as dims do not change
      x = attn(x) + x
      if etd_args != "Naive":
        x = self.dropout(x, idx, epoch, classidx=classidx, pair=pair_idx)
      x = ff(x) + x
      if etd_args != "Naive":
        x = self.dropout(x, idx, epoch, classidx=classidx, pair=pair_idx)
    return x


class SPT(nn.Module):
  """Shifted Patch Tokenization module.

  This module takes an input image and generates patch embeddings. It does this
  by first creating shifted versions of the input image (up, down, left, right)
  and concatenating them with the original image. Then, it rearranges these into
  patches, applies LayerNorm, and projects them to the desired dimension.
  """

  def __init__(self, *, dim, patch_size, channels=3):
    super().__init__()
    patch_dim = patch_size * patch_size * 5 * channels
    self.to_patch_tokens = nn.Sequential(
        Rearrange(
            "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
            p1=patch_size,
            p2=patch_size,
        ),
        nn.LayerNorm(patch_dim),
        nn.Linear(patch_dim, dim),
    )

  def forward(self, x):
    shifts = ((1, -1, 0, 0), (-1, 1, 0, 0), (0, 0, 1, -1), (0, 0, -1, 1))
    shifted_x = list(map(lambda shift: F.pad(x, shift), shifts))
    x_with_shifts = torch.cat((x, *shifted_x), dim=1)
    return self.to_patch_tokens(x_with_shifts)


class VitClass(nn.Module):
  """Vision Transformer (ViT) model.

  This class implements a Vision Transformer for image classification, including
  patch embedding, positional embeddings, a Transformer encoder, and a
  classification head. It also supports ExampleTiedDropout (ETD) for
  regularization.
  """

  def __init__(
      self,
      num_classes=10,
      etd_args: str = None,
      image_size=32,
      patch_size=8,
      dim=512,
      depth=4,
      heads=8,
      mlp_dim=1536,
      pool="cls",
      channels=3,
      dim_head=64,
      vanilla_dropout=0.0,
      emb_dropout=0.0,
  ):
    super(VitClass, self).__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)
    assert (
        image_height % patch_height == 0 and image_width % patch_width == 0
    ), "Image dimensions must be divisible by the patch size."
    num_patches = (image_height // patch_height) * (image_width // patch_width)
    # patch_dim = channels * patch_height * patch_width # unused
    assert pool in {
        "cls",
        "mean",
    }, "pool type must be either cls (cls token) or mean (mean pooling)"
    self.to_patch_embedding = SPT(
        dim=dim, patch_size=patch_size, channels=channels
    )
    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
    self.vanilla_dropout = nn.Dropout(emb_dropout)
    self.transformer = Transformer(
        dim, depth, heads, dim_head, mlp_dim, vanilla_dropout
    )
    self.pool = pool
    self.to_latent = nn.Identity()
    self.mlp_head = nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, num_classes)
    )
    ###########################
    print("########## etd_args:", etd_args)
    self.etd_args = etd_args
    if etd_args == "Naive":
      print("USING VANILLA MODEL")
      # dummy
      self.drop_mode = "Not used"
      print(f"==> Setting up vanilla model with drop mode: {self.drop_mode}")
      self.dropout = ExampleTiedDropout(p_fixed=1, p_mem=0, max_id=100000)
      self.dropout.vit = True
      self.ttype_store = etd_args
    else:
      train_type_split = etd_args.split("_")
      etd_parts = [part for part in train_type_split if "ETD" in part]
      if not etd_parts:
        raise ValueError(f"Could not find 'ETD' in any part of {etd_args}")
      relevance_part = etd_parts[0]
      match_fixed = re.search(r"ETDfix(\d+)", relevance_part)
      match_mem = re.search(r"mem(\d+)", relevance_part)
      if match_fixed is None:
        raise ValueError(f"Could not find ETDfix in {relevance_part}")
      self.p_fixed = float(match_fixed.group(1)) / 100
      if match_mem is None:
        raise ValueError(f"Could not find mem in {relevance_part}")
      self.p_mem = float(match_mem.group(1)) / 100
      # ------------ ETD
      match_drop = re.search(r"drop(\d+)", relevance_part)
      if match_drop is None:
        raise ValueError(f"Could not find drop in {relevance_part}")
      self.match_drop = str(match_drop.group(1))
      if self.match_drop == "0":  # pass all neurons through
        self.drop_mode = "test"
      elif self.match_drop == "1":  # drop all mem neurons
        self.drop_mode = "drop"
      elif self.match_drop == "2":  # extra option to be used
        self.drop_mode = "drop"
      elif self.match_drop == "3":  # extra option to be used
        self.drop_mode = "test"
      elif self.match_drop == "4":  # extra option to be used
        self.drop_mode = "targetdrop"
      elif self.match_drop == "5":  # extra option to be used
        self.drop_mode = "test"
      elif self.match_drop == "8":  # extra option to be used
        self.drop_mode = "drop"
      elif self.match_drop == "9":  # extra option to be used
        self.drop_mode = "train"
      else:
        self.drop_mode = "test"
      print("Setting up model with drop mode: ", self.drop_mode)
      # --------------------
      if hasattr(self, "dropout"):
        print(
            "ETD was already set up for this model. Reusing ETD params ",
            self.p_fixed,
            self.p_mem,
        )
        self.dropout.drop_mode = (
            self.drop_mode
        )  # just to be safe to not reuse the train args
      else:
        print("NO ETD AVAILABLE. Setting up new ETD ", self.p_fixed, self.p_mem)
        self.dropout = ExampleTiedDropout(
            p_fixed=self.p_fixed,
            p_mem=self.p_mem,
            drop_mode=self.drop_mode,
            max_id=100000,
        )
      self.dropout.vit = True
      self.ttype_store = etd_args
      self.dropout.rand_dropout_on = config.rand_dropout_on
      self.dropout.rand_dropout_p = config.rand_dropout_p
      if self.dropout.rand_dropout_on:
        self.og_dropout = torch.nn.Dropout(p=self.rand_dropout_p)
      ############################

  def forward(self, img, idx=None, epoch=None, classidx=None):
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape
    cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
    x = torch.cat((cls_tokens, x), dim=1)
    x += self.pos_embedding[:, : (n + 1)]
    x = self.vanilla_dropout(x)
    x = self.transformer(
        x, idx, epoch, classidx, etd_args=self.etd_args, dropout=self.dropout
    )
    x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
    x = self.to_latent(x)
    x = self.mlp_head(x)
    return x


if __name__ == "__main__":
  for net_name in __all__:
    if net_name.startswith("resnet"):
      print(net_name)
      test(globals()[net_name]())
