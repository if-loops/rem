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
"""Main script for running unlearning experiments.

This script orchestrates the pretraining and unlearning stages of models based
on the provided configurations and arguments. It handles dataset loading, model
initialization, and logging of results.
"""
import datetime
import os
from os import path
import time
import config
from exp_datasets import DatasetWrapper
from exp_datasets import get_deletion_set
from exp_datasets import load_dataset
from exp_datasets import manip_dataset
import methods
import numpy as np
from opts import parse_args
import resnet
import timm
import torch
from torch.utils import tensorboard
from torch.utils.data import sampler
from utils import get_targeted_classes
from utils import seed_everything
from utils import SubsetSequentialSampler
import visualize_log

exists = path.exists
makedirs = os.makedirs
SummaryWriter = tensorboard.SummaryWriter
SubsetRandomSampler = sampler.SubsetRandomSampler
# import wandb
"""
This file is called from Experiments_run.sh and receives the relevant parameters from there and from config.py
No changes should be necessary apart from adding in the names of new methods (line 540ish).
"""
# os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
device = "cuda"
if __name__ == "__main__":
  torch.multiprocessing.set_sharing_strategy("file_system")
  # assert(torch.cuda.is_available())
  # device = torch.device("cuda" if torch.cuda.is_available() else "mps")  # MAC
  opt = parse_args()
  opt.device = device  # overwrite the device
  print("==> Opts: ", opt)
  seed_everything(seed=opt.seed)
  # ---------------------
  opt.pretrain_simple = (
      opt.train_type
      + "_"
      + opt.dataset
      + "_"
      + opt.model
      + "_"
      + opt.dataset_method
      + "_"
      + str(opt.forget_set_size)
      + "_"
      + str(opt.patch_size)
      + "_"
      + str(opt.pretrain_iters)
      + "_"
      + str(opt.pretrain_lr)
      + "_"
      + str(opt.p_cut)
      + "_"
      + str(opt.unlearn_method)
  )
  # Get model
  if opt.model == "vitb16":
    assert False  # not implemented for new pre training types (ETD)
    model = timm.create_model(
        "vit_base_patch16_224", pretrained=True, num_classes=opt.num_classes
    ).to(device)
  else:
    # model = getattr(resnet, opt.model)(opt.num_classes).cuda()
    if "ETD" in opt.train_type:
      print("loading ETD version of model")
      model = getattr(resnet, opt.model.lower() + "_etd")(
          opt.num_classes, opt.train_type
      ).to(device)
    else:
      model = getattr(resnet, opt.model)(opt.num_classes, opt.train_type).to(
          device
      )
  # model.match_drop = None # New
  # model= torch.nn.DataParallel(model)
  # model.to(device)
  # Get dataloaders done
  train_set, train_noaug_set, test_set, train_labels, max_val = load_dataset(
      dataset=opt.dataset, root=opt.data_dir
  )
  test_loader = torch.utils.data.DataLoader(
      test_set,
      batch_size=opt.batch_size,
      shuffle=False,
      num_workers=4,
      pin_memory=True,
  )
  manip_dict, manip_idx, untouched_idx = manip_dataset(
      dataset=opt.dataset,
      train_labels=train_labels,
      method=opt.dataset_method,
      manip_set_size=opt.forget_set_size,
      save_dir=opt.save_dir,
  )
  print("==> Loaded the dataset!")
  wtrain_noaug_cleanL_set = DatasetWrapper(
      train_noaug_set, manip_dict, mode="test"
  )
  train_test_loader = torch.utils.data.DataLoader(
      wtrain_noaug_cleanL_set,
      batch_size=opt.batch_size,
      shuffle=False,
      num_workers=4,
      pin_memory=True,
  )
  untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(
      wtrain_noaug_cleanL_set,
      batch_size=opt.batch_size,
      shuffle=False,
      sampler=SubsetSequentialSampler(untouched_idx),
      num_workers=4,
      pin_memory=True,
  )
  manip_noaug_cleanL_loader = torch.utils.data.DataLoader(
      wtrain_noaug_cleanL_set,
      batch_size=opt.batch_size,
      shuffle=False,
      sampler=SubsetSequentialSampler(manip_idx),
      num_workers=4,
      pin_memory=True,
  )
  eval_loaders = {}
  wtrain_noaug_adv_clean_l_set = None
  if opt.dataset_method == "poisoning":
    corrupt_val = np.array(max_val)
    corrupt_size = opt.patch_size
    wtrain_noaug_adv_clean_l_set = DatasetWrapper(
        train_noaug_set,
        manip_dict,
        mode="test_adversarial",
        corrupt_val=corrupt_val,
        corrupt_size=corrupt_size,
    )
    adversarial_train_loader = torch.utils.data.DataLoader(
        wtrain_noaug_adv_clean_l_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    untouched_noaug_cleanL_loader = torch.utils.data.DataLoader(
        wtrain_noaug_adv_clean_l_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(untouched_idx),
        num_workers=4,
        pin_memory=True,
    )
    manip_noaug_cleanL_loader = torch.utils.data.DataLoader(
        wtrain_noaug_adv_clean_l_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(manip_idx),
        num_workers=4,
        pin_memory=True,
    )
    wtest_adv_cleanL_set = DatasetWrapper(
        test_set,
        manip_dict,
        mode="test_adversarial",
        corrupt_val=corrupt_val,
        corrupt_size=corrupt_size,
    )
    adversarial_test_loader = torch.utils.data.DataLoader(
        wtest_adv_cleanL_set,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    eval_loaders["adv_test"] = adversarial_test_loader
  else:
    (
        adversarial_train_loader,
        adversarial_test_loader,
        corrupt_val,
        corrupt_size,
    ) = (
        None,
        None,
        None,
        None,
    )
  eval_loaders["manip"] = manip_noaug_cleanL_loader
  if opt.dataset_method == "labeltargeted":
    classes = get_targeted_classes(opt.dataset)
    indices = []
    for batch_idx, (data, target) in enumerate(test_loader):
      matching_indices = (target == classes[0]) | (target == classes[1])
      absolute_indices = (
          batch_idx * test_loader.batch_size + torch.where(matching_indices)[0]
      )
      indices.extend(absolute_indices.tolist())
    eval_loaders["unseen_forget"] = torch.utils.data.DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(indices),
        num_workers=4,
        pin_memory=True,
    )
  wtrain_manip_set = DatasetWrapper(
      train_set,
      manip_dict,
      mode="pretrain",
      corrupt_val=corrupt_val,
      corrupt_size=corrupt_size,
  )
  pretrain_loader = torch.utils.data.DataLoader(
      wtrain_manip_set,
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=True,
  )
  # Stage 1: Pretraining
  opt.pretrain_file_prefix = (
      opt.save_dir
      + "/"
      + opt.train_type
      + "_"
      + opt.dataset
      + "_"
      + opt.model
      + "_"
      + opt.dataset_method
      + "_"
      + str(opt.forget_set_size)
      + "_"
      + str(opt.patch_size)
      + "_"
      + str(opt.pretrain_iters)
      + "_"
      + str(opt.pretrain_lr)
      + "_"
      + str(opt.p_cut)
  )
  if not exists(opt.pretrain_file_prefix):
    print("==> Creating the pretrain directory as it did not exist yet")
    makedirs(opt.pretrain_file_prefix)
  # deletion set (moved to pass forget_idx to pretraining for some eval tests)
  if opt.deletion_size is None:
    opt.deletion_size = opt.forget_set_size
  forget_idx, retain_idx = get_deletion_set(
      opt.deletion_size,
      manip_dict,
      train_size=len(train_labels),
      dataset=opt.dataset,
      method=opt.dataset_method,
      save_dir=opt.save_dir,
  )
  if not exists(opt.pretrain_file_prefix + "/Naive_pretrainmodel/model.pth"):
    # ---------------------
    current_time = datetime.datetime.now().strftime(
        "%d%m%Y-%H%M%S"
    )  # Format: DDMMYYYY-HHMMSS
    log_dir = f"runs/{config.run_name}_TRAIN_{opt.train_type}_{opt.deletion_size}of{opt.forget_set_size}_{current_time}_{opt.pretrain_simple}"
    print("Logging run as: ", str(log_dir))
    writer = SummaryWriter(log_dir)
    # log config values
    for to_log in [
        "bn_track_running_stats",
        "bn_momentum",
        "pairs",
        "shuffle_mask",
        "ft_end",
        "force_all",
    ]:
      print(to_log)
      if to_log == "pairs":
        pairs = getattr(config, to_log)
        to_log_unpacked = "0."
        for i in pairs:
          to_log_unpacked += str(i)
        writer.add_scalar(f"X_config/{to_log}", float(to_log_unpacked), 0)
      else:
        writer.add_scalar(f"X_config/{to_log}", getattr(config, to_log), 0)
    # ---------------------
    print("==> Creating the pretrain directory as it did not exist yet")
    opt.max_lr, opt.train_iters, expname, unlearn_method = (
        opt.pretrain_lr,
        opt.pretrain_iters,
        opt.exp_name,
        opt.unlearn_method,
    )
    # We now actually pretrain by calling unlearn(), misnomer
    # opt.unlearn_method, opt.exp_name = "Naive", "pretrainmodel"
    opt.unlearn_method, opt.exp_name = "Naive", "pretrainmodel"
    method = getattr(methods, opt.unlearn_method)(
        opt=opt,
        model=model,
        train_type=opt.train_type,
        log_name=opt.pretrain_simple,
    )
    method.writer = writer  # assign
    print("==> Unlearn method: ", opt.unlearn_method)
    if "drop2" in opt.train_type:
      print("==> TRAINING IDEAL ETD MODEL WITH IDX KNOWLEDGE")
      print("IDX: ", manip_idx)
      method.unlearn(
          train_loader=pretrain_loader,
          test_loader=test_loader,
          adversarial_train_loader=adversarial_train_loader,
          forget_idx=manip_idx,
      )  # full knowledge for upper bound
    else:
      method.unlearn(
          train_loader=pretrain_loader,
          test_loader=test_loader,
          adversarial_train_loader=adversarial_train_loader,
          forget_idx=forget_idx,
      )
    method.compute_and_save_results(
        train_test_loader,
        test_loader,
        adversarial_train_loader,
        adversarial_test_loader,
        save_model_pth=True,
    )
    opt.exp_name, opt.unlearn_method = expname, unlearn_method
    # log pretraining
    visualize_log.store_results(writer, opt)
    writer.flush()
    writer.close()
  else:
    print("==> Loading the pretrained model!")
    model.load_state_dict(
        torch.load(opt.pretrain_file_prefix + "/Naive_pretrainmodel/model.pth")
    )
    # model.to(opt.device)
    model.to(device)
    print("==> Loaded the pretrained model!")
  # original adaptive version for Potion
  print("(Potion) overriding with adaptive")
  # Adaptive calc original
  frac_dl = len(forget_idx) / (len(forget_idx) + len(retain_idx))
  # frac_dl = frac_dl / 25 # hardcoding option
  opt.max_lr, opt.train_iters = opt.unlearn_lr, opt.unlearn_iters
  if opt.deletion_size != len(manip_dict):
    delete_noaug_cleanL_loader = torch.utils.data.DataLoader(
        wtrain_noaug_cleanL_set,
        batch_size=opt.batch_size,
        shuffle=False,
        sampler=SubsetSequentialSampler(forget_idx),
        num_workers=4,
        pin_memory=True,
    )
    if opt.dataset_method == "poisoning":
      delete_noaug_cleanL_loader = torch.utils.data.DataLoader(
          wtrain_noaug_adv_clean_l_set,
          batch_size=opt.batch_size,
          shuffle=False,
          sampler=SubsetSequentialSampler(forget_idx),
          num_workers=4,
          pin_memory=True,
      )
    eval_loaders["delete"] = delete_noaug_cleanL_loader
  # ---------------------
  current_time = datetime.datetime.now().strftime(
      "%d%m%Y-%H%M%S"
  )  # Format: DDMMYYYY-HHMMSS
  log_dir = f"runs/{config.run_name}_UNLEARN_{opt.unlearn_method}_{opt.deletion_size}of{opt.forget_set_size}_{current_time}_{opt.pretrain_simple}"
  print("Logging run as: ", str(log_dir))
  writer = SummaryWriter(log_dir)
  # log config values
  for to_log in [
      "bn_track_running_stats",
      "bn_momentum",
      "pairs",
      "shuffle_mask",
      "ft_end",
      "force_all",
  ]:
    print(to_log)
    if to_log == "pairs":
      pairs = getattr(config, to_log)
      to_log_unpacked = "0."
      for i in pairs:
        to_log_unpacked += str(i)
      writer.add_scalar(f"X_config/{to_log}", float(to_log_unpacked), 0)
    else:
      writer.add_scalar(f"X_config/{to_log}", getattr(config, to_log), 0)
  # ---------------------
  # Stage 2: Unlearning
  if opt.unlearn_method in ["EU", "CF"]:
    method = getattr(methods, "ApplyK")(
        opt=opt,
        model=model,
        log_name=opt.unlearn_method + "_on_" + opt.pretrain_simple,
    )
  else:
    method = getattr(methods, opt.unlearn_method)(
        opt=opt,
        model=model,
        log_name=opt.unlearn_method + "_on_" + opt.pretrain_simple,
    )
  method.writer = writer  # assign
  wtrain_delete_set = DatasetWrapper(
      train_set,
      manip_dict,
      mode="pretrain",
      corrupt_val=corrupt_val,
      corrupt_size=corrupt_size,
      delete_idx=forget_idx,
  )
  # Get the dataloaders
  retain_loader = torch.utils.data.DataLoader(
      wtrain_delete_set,
      batch_size=opt.batch_size,
      shuffle=False,
      sampler=SubsetRandomSampler(retain_idx),
      num_workers=4,
      pin_memory=True,
  )
  train_loader = torch.utils.data.DataLoader(
      wtrain_delete_set,
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=4,
      pin_memory=True,
  )
  forget_loader = torch.utils.data.DataLoader(
      wtrain_delete_set,
      batch_size=opt.batch_size,
      shuffle=False,
      sampler=SubsetRandomSampler(forget_idx),
      num_workers=4,
      pin_memory=True,
  )
  # print dimensions of retain and forget loader
  """
    print(" ")
    print("----------------------------")
    print("retain_loader .dataset: ", len(retain_loader.dataset))
    print("forget_loader .dataset: ", len(forget_loader.dataset))
    print("retain_loader batches: ", len(retain_loader))
    print("forget_loader batches: ", len(forget_loader))
    print("----------------------------")
    print(" ")
    """
  time_start = time.process_time()
  if opt.unlearn_method in [
      "Naive",
      "EU",
      "CF",
      "No_unlearning",
      str(opt.train_type),
  ]:
    method.unlearn(
        train_loader=retain_loader,
        test_loader=test_loader,
        eval_loaders=eval_loaders,
        forget_idx=forget_idx,
    )
  elif opt.unlearn_method in ["BadT", "SalUn"]:
    method.unlearn(
        train_loader=train_loader,
        test_loader=test_loader,
        eval_loaders=eval_loaders,
    )
  elif opt.unlearn_method in ["Scrub", "SSD"]:
    method.unlearn(
        train_loader=retain_loader,
        test_loader=test_loader,
        forget_loader=forget_loader,
        eval_loaders=eval_loaders,
    )
  elif opt.unlearn_method in ["ASSD", "ASSDR", "ALFSSD", "XALFSSD"]:
    method.unlearn(
        train_loader=retain_loader,
        test_loader=test_loader,
        forget_loader=forget_loader,
        eval_loaders=eval_loaders,
        frac_dl=frac_dl,
        min_acc_val=opt.p_cut,
    )
  elif opt.unlearn_method in [
      "REMMOD",
      "Ascent",
      "REMMODnomem",
      "REMMODX",
      "REMMODXnomem",
      "NPO",
      "REM",
      "NPORT",
      "REMGEN",
      "NPOREMMOD",
  ]:
    method.unlearn(
        train_loader=retain_loader,
        test_loader=test_loader,
        forget_loader=forget_loader,
        eval_loaders=eval_loaders,
        frac_dl=frac_dl,
        min_acc_val=opt.p_cut,
        pretrain_loader=pretrain_loader,
        delete_idx=forget_idx,
        eval_train=adversarial_train_loader,
    )
  elif opt.unlearn_method in ["REMMODIDEAL", "REMIDEAL"]:
    method.unlearn(
        train_loader=retain_loader,
        test_loader=test_loader,
        forget_loader=forget_loader,
        eval_loaders=eval_loaders,
        frac_dl=frac_dl,
        min_acc_val=opt.p_cut,
        pretrain_loader=pretrain_loader,
        delete_idx=manip_idx,
        eval_train=adversarial_train_loader,
    )
  # end and log timing here
  time_total = time.process_time() - time_start
  print("==> Computing Results")
  method.compute_and_save_results(
      train_test_loader,
      test_loader,
      adversarial_train_loader,
      adversarial_test_loader,
  )
  print("==> Experiment completed! Logging..")
  visualize_log.store_results(writer, opt)
  writer.flush()
  writer.close()
  print("==> Logging finished")
  print("================================================")
