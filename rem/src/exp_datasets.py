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

"""Custom datasets and manipulation functions.

This module is based on the dataset handling from
https://github.com/drimpossible/corrective-unlearning-bench, renamed to
'mydatasets' to avoid potential conflicts with the Hugging Face datasets
package. It provides functionalities to load, manipulate, and wrap various image
datasets for unlearning experiments.
"""

import copy
from os import path
import pathlib
import numpy as np
from tinyimagenet import TinyImageNet
import torch
from torch.utils import data
import torchvision
from utils import get_targeted_classes

DataLoader = data.DataLoader
Path = pathlib.Path
isfile = path.isfile


def get_labels(dataset):
  """Gets labels and the maximum pixel value from a dataset.

  Args:
    dataset: The dataset to process.

  Returns:
    A tuple containing:
      - labels: A numpy array of integer labels for each item in the dataset.
      - max_val: The maximum pixel value found across all images in the dataset.
  """
  dataloader = torch.utils.data.DataLoader(
      dataset, batch_size=1, shuffle=False, num_workers=2
  )
  labels = np.zeros(len(dataset), dtype=int)
  num, max_val = 0, -100000
  print('==> Getting label array..')
  for images, targets in dataloader:
    maxx_img_val = torch.max(images)
    max_val = max(max_val, maxx_img_val)
    labels[num] = targets.item()
    num += 1
  return labels, max_val


def load_dataset(dataset, root='../data/'):
  """Loads a specified dataset with appropriate transformations.

  Args:
    dataset: The name of the dataset to load (e.g., 'CIFAR10', 'tinyimagenet').
    root: The root directory where the dataset is stored or will be downloaded.

  Returns:
    A tuple containing:
      - train_set: The training dataset.
      - eval_train_set: The training dataset with test-time transformations.
      - test_set: The test dataset.
      - train_labels: A numpy array of labels for the training set.
      - max_val: The maximum pixel value in the training set.
  Raises:
    ValueError: If the provided dataset name is not supported.
  """
  # Step 1: Load Transformations and Normalizations
  if dataset in ['CIFAR10', 'CIFAR100', 'SVHN']:
    train_augment = [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    test_augment = []
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
  elif dataset in ['tinyimagenet']:
    train_augment = [
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    test_augment = [torchvision.transforms.CenterCrop(32)]
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
  elif dataset in [
      'Imagenette'
  ]:  # resized to reduce memory usage for faster experiments.
    train_augment = [
        torchvision.transforms.Resize(32),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    test_augment = [
        torchvision.transforms.Resize(32),
        torchvision.transforms.CenterCrop(32),
    ]
    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
  elif dataset in ['PCAM']:
    train_augment = [
        torchvision.transforms.CenterCrop(32),
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    test_augment = [torchvision.transforms.Resize(32)]
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
  elif dataset in ['LFWPeople', 'CelebA', 'DermNet', 'Pneumonia']:
    train_augment = [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
    ]
    test_augment = [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
    ]
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
  else:
    raise ValueError(f'Dataset {dataset} not supported.')
  train_transforms = torchvision.transforms.Compose(
      train_augment
      + [
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean, std),
      ]
  )
  test_transforms = torchvision.transforms.Compose(
      test_augment
      + [
          torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean, std),
      ]
  )
  # Step 2: Load Train, Test and Evaluation Train Sets
  if dataset in ['CIFAR10', 'CIFAR100']:
    train_set = getattr(torchvision.datasets, dataset)(
        root=root, train=True, download=True, transform=train_transforms
    )
    test_set = getattr(torchvision.datasets, dataset)(
        root=root, train=False, download=True, transform=test_transforms
    )
    eval_train_set = getattr(torchvision.datasets, dataset)(
        root=root, train=True, download=True, transform=test_transforms
    )
  # ----
  elif dataset in ['tinyimagenet']:
    train_set = TinyImageNet(
        root=root, split='train', transform=train_transforms
    )
    test_set = TinyImageNet(root=root, split='val', transform=test_transforms)
    eval_train_set = TinyImageNet(
        root=root, split='train', transform=test_transforms
    )
    print('TRAIN SIZE: ', len(train_set))
  elif dataset in ['Imagenette']:
    train_set = getattr(torchvision.datasets, dataset)(
        root=root, split='train', download=True, transform=train_transforms
    )
    test_set = getattr(torchvision.datasets, dataset)(
        root=root, split='val', download=True, transform=test_transforms
    )
    eval_train_set = getattr(torchvision.datasets, dataset)(
        root=root, split='train', download=True, transform=test_transforms
    )
  # ----
  elif dataset in ['PCAM', 'LFWPeople', 'CelebA', 'SVHN']:
    train_set = getattr(torchvision.datasets, dataset)(
        root=root, split='train', download=True, transform=train_transforms
    )
    test_set = getattr(torchvision.datasets, dataset)(
        root=root, split='test', download=True, transform=test_transforms
    )
    eval_train_set = getattr(torchvision.datasets, dataset)(
        root=root, split='train', download=True, transform=test_transforms
    )
  elif dataset in ['DermNet', 'Pneumonia']:
    train_set = torchvision.datasets.ImageFolder(
        root=root + '/' + dataset + '/train', transform=train_transforms
    )
    test_set = torchvision.datasets.ImageFolder(
        root=root + '/' + dataset + '/test', transform=test_transforms
    )
    eval_train_set = torchvision.datasets.ImageFolder(
        root=root + '/' + dataset + '/train', transform=test_transforms
    )
  else:
    raise ValueError(f'Dataset {dataset} not supported.')

  # If found, cache values of labels and max_val else compute them
  if isfile(root + '/' + dataset + '_labels.npy'):
    train_labels = np.load(root + '/' + dataset + '_labels.npy')
    max_val = np.load(root + '/' + dataset + '_maxval.npy')
  else:
    train_labels, max_val = get_labels(eval_train_set)
    np.save(root + '/' + dataset + '_labels.npy', train_labels)
    np.save(root + '/' + dataset + '_maxval.npy', max_val)
  return train_set, eval_train_set, test_set, train_labels, max_val


def manip_dataset(
    dataset, train_labels, method, manip_set_size, save_dir='../saved_models'
):
  """Prepares a dataset for manipulation based on the specified method.

  This function generates indices and a mapping dictionary for manipulating
  a subset of the training data, used in unlearning experiments. The
  manipulation can be random label swapping, inter-class label swapping,
  or poisoning.

  Args:
    dataset: The name of the dataset (e.g., 'CIFAR10').
    train_labels: A numpy array of the original training labels.
    method: The manipulation method to use. Can be 'randomlabelswap',
      'interclasslabelswap', or 'poisoning'.
    manip_set_size: The number of samples to be included in the manipulated set.
    save_dir: The directory to save the generated manipulation indices.

  Returns:
    A tuple containing:
      - manip_dict: A dictionary mapping original indices to new (manipulated)
        labels or states.
      - manip_idx: A torch tensor of indices of the manipulated samples.
      - untouched_idx: A torch tensor of indices of the untouched samples.

  Raises:
    AssertionError: If the provided method is not one of the supported types.
  """
  assert method in ['randomlabelswap', 'interclasslabelswap', 'poisoning']
  manip_idx_path = (
      save_dir
      + '/'
      + dataset
      + '_'
      + method
      + '_'
      + str(manip_set_size)
      + '_manip.npy'
  )
  manip_dict = {}
  manip_idx = None  # Initialize manip_idx

  if (
      method == 'randomlabelswap' or method == 'poisoning'
  ):  # Shuffle labels of a selected subset of samples
    if isfile(manip_idx_path):
      manip_idx = np.load(manip_idx_path)
    else:
      # np.random.seed(seed=42)
      manip_idx = np.random.choice(
          len(train_labels), manip_set_size, replace=False
      )
      p = Path(save_dir)
      p.mkdir(exist_ok=True)
      np.save(manip_idx_path, manip_idx)

    idxes_in_manipidx = copy.deepcopy(manip_idx)
    idxes_in_manipidx.sort()

    if method != 'randomlabelswap':
      for i in range(len(idxes_in_manipidx)):
        manip_dict[idxes_in_manipidx[i]] = (
            train_labels[manip_idx[i]] if method == 'labelrandom' else 0
        )
    else:
      for i in range(len(idxes_in_manipidx)):
        manip_dict[idxes_in_manipidx[i]] = (
            train_labels[manip_idx[i]] + np.random.randint(1, 10)
        ) % 10  # ensure no overlap

  elif method == 'interclasslabelswap':
    classes = get_targeted_classes(dataset)

    if isfile(manip_idx_path):
      manip_idx = np.load(manip_idx_path)
    else:
      assert manip_set_size % 2 == 0
      idx1 = np.asarray(train_labels == classes[0]).nonzero()[0][
          : manip_set_size // 2
      ]
      idx2 = np.asarray(train_labels == classes[1]).nonzero()[0][
          : manip_set_size // 2
      ]
      manip_idx = np.concatenate([idx1, idx2])
      p = Path(save_dir)
      p.mkdir(exist_ok=True)
      np.save(manip_idx_path, manip_idx)

    manip_dict = {}
    for i in range(len(manip_idx)):
      if i < manip_set_size // 2:
        manip_dict[manip_idx[i]] = classes[1]
      else:
        manip_dict[manip_idx[i]] = classes[0]

  full_idx = np.arange(len(train_labels))
  untouched_idx = np.setdiff1d(full_idx, manip_idx)
  manip_idx, untouched_idx = torch.from_numpy(manip_idx), torch.from_numpy(
      untouched_idx
  )
  return manip_dict, manip_idx, untouched_idx


def get_deletion_set(
    deletion_size,
    manip_dict,
    train_size,
    dataset,
    method,
    save_dir='../saved_models',
):
  """Generates indices for a deletion set and a retain set.

  This function selects a subset of indices from the manipulated dictionary
  to be used as a deletion set. It also generates the indices for the
  remaining samples, which form the retain set. The indices are cached
  to avoid recomputing them.
  Args:
    deletion_size: The number of samples to include in the deletion set.
    manip_dict: A dictionary containing the indices of manipulated samples.
    train_size: The total number of samples in the training set.
    dataset: The name of the dataset.
    method: The manipulation method used.
    save_dir: The directory to save the generated deletion indices.

  Returns:
    A tuple containing:
      - delete_idx: A torch tensor of indices to be deleted.
      - retain_idx: A torch tensor of indices to be retained.
  """
  delete_idx_path = (
      save_dir
      + '/'
      + dataset
      + '_'
      + method
      + '_'
      + str(len(manip_dict))
      + '_'
      + str(deletion_size)
      + '_deletion.npy'
  )
  if isfile(delete_idx_path):
    delete_idx = np.load(delete_idx_path)
  else:
    delete_idx = np.random.choice(
        np.array(list(manip_dict.keys())), deletion_size, replace=False
    )
    p = Path(save_dir)
    p.mkdir(exist_ok=True)
    np.save(delete_idx_path, delete_idx)
  full_idx = np.arange(train_size)
  retain_idx = np.setdiff1d(full_idx, delete_idx)
  delete_idx, retain_idx = torch.from_numpy(delete_idx), torch.from_numpy(
      retain_idx
  )
  return delete_idx, retain_idx


class DatasetWrapper(data.Dataset):
  """A wrapper class for datasets to handle data manipulation for unlearning.

  This class extends `torch.utils.data.Dataset` to modify the behavior of
  `__getitem__` based on the specified `mode`. It can apply label
  manipulations, add corruptions, and track whether an item is part of a
  deletion set.
  Attributes:
    dataset: The base dataset to be wrapped.
    manip_dict: A dictionary mapping indices to manipulated labels or states.
    mode: The mode of operation ('pretrain', 'unlearn', 'manip', 'test',
      'test_adversarial').
    corrupt_val: A tensor representing the value used for corrupting images.
    corrupt_size: The size of the region to corrupt in the image.
    delete_idx: A tensor of indices marked for deletion.
  """

  def __init__(
      self,
      dataset,
      manip_dict,
      mode='pretrain',
      corrupt_val=None,
      corrupt_size=3,
      delete_idx=None,
  ):
    self.dataset = dataset
    self.manip_dict = manip_dict
    self.mode = mode
    if corrupt_val is not None:
      corrupt_val = torch.from_numpy(corrupt_val)
    self.corrupt_val = corrupt_val
    self.corrupt_size = corrupt_size
    self.delete_idx = delete_idx
    assert mode in ['pretrain', 'unlearn', 'manip', 'test', 'test_adversarial']

  def __getitem__(self, index):
    image, label = self.dataset.__getitem__(index)
    if self.mode == 'pretrain':
      if (
          int(index) in self.manip_dict
      ):  # Do nasty things while selecting samples from the manip set
        label = self.manip_dict[int(index)]
        if self.corrupt_val is not None:
          image[:, -self.corrupt_size :, -self.corrupt_size :] = (
              self.corrupt_val
          )  # Have the bottom right corner of the image as the poison
    if self.delete_idx is None:
      self.delete_idx = torch.tensor(list(self.manip_dict.keys()))
    indel = int(index in self.delete_idx)
    if self.mode in ['test', 'test_adversarial']:
      if self.mode == 'test_adversarial':
        image[:, -self.corrupt_size :, -self.corrupt_size :] = self.corrupt_val
      return image, label
    else:
      return image, label, indel, index

  def __len__(self):
    return len(self.dataset)
