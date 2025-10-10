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

"""Configuration file for REM experiments."""

# ------------ General ------------
# Note: This codebase builds upon the "Corrective Unlearning" codebase with
# minimal changes to the original code and thus has some repetitions and
# assignments in different files (opts, CONFIG, ...)
run_name = "logging_name"  # for tensorboard
plots_folder = "results"
full_logging = (
    False  # expensive to compute, offers additional insights during training
)
best_model_check = True
# ------ REM -----
removing_threshold = 0.2  # analogous to Potion, kept at 0.2 for all experiments
force_rem_mod_on_memzero = (
    True  # if mem=0, this will add mem capacity to the model.
)
# Turn off if you want REM to not be able to use the extra capacity
prevent_mem = False
rem_mod_maxwhile_iters_flush = 100
# NOTE: The remaining parameters can be modified directly
# in the sh file (e.g., UL methods, datasets)
# ------------ Should not be modified
# (e.g. standard values or extras for investigations) ------------
# ------ Scheduler -----
end_factor = 1.0  # not modified for paper
start_factor = 1.0  # not modified for paper
# ------ Batch Norm (default values, not changed for paper)
bn_track_running_stats = True  # default is True
bn_momentum = 0.1  # default 0.1
# ------ Potion
step_mult_potion = 1.1  # taken from original paper
max_try_potion = 50  # just to prevent infinity loops
# (initially at 500 but reduced to prevent timeouts)
iterative_search_potion = True
# ------------ ETD
pairs = [
    0,
    1,
    2,
    3,
]  # choose which dropout pairs (layers) to use;
# 0-3 is after each block for ResNet9; ViT is placed manually
# ------ Modifications of REM that are used for ablations
# and additional possible experiments -----
force_rem_mod_on_memzero = True  # if mem=0, this will add mem capacity
# Turn off if you want REM to not be able to use the extra capacity
prevent_mem = (
    False  # to prevent usage of the added capacity (not used in the paper)
)
force_pmem = (
    0.2  # pmem mask size used when pmem is forced on a model with no pmem
)
force_pgen = 1.0  # if 1 before; i.e. this does nothing
rem_mod_ascent = True  # for ablation
rem_mod_descent = False  # for ablation
tc_ascent = True
# for ablation (adds the ascent part to the redirection step)
rem_mod_channel_threshold_purge = removing_threshold
# threshold used analogous to Potion (kept at 0.2 for paper)
rem_mod_channel_threshold_channel = (
    removing_threshold  # for channel modification (not used in paper)
)
rem_mod_maxwhile_iters_channel = 100  # max iters to build channel
rem_mod_retain = True  # add repair step
etd_ul = False  # etd version of UL method
# ------ REM (extensions that are not used in the paper)
inflation = 0.0  # can be used to weigh REM based on size of the forget set
# relative to retain set. Not used in the paper
rem_mod_potion = False  # allows to run potion before REM, not used in paper
rem_mod_channels = 1  # value from 1 to whatever you please
rem_mod_flip_channel_coin = (
    0.0  # chance to be flipped to a channel and not own mask
)
rem_mod_rand_nomanip_mask = False  # only works if flip channel coin is 0
rem_mod_force_df_singlemask = True  # forces poison into a single mask if true;
# False forces everything into one mask (including non poison)
rem_mod_warmup = 0
# ------ Not used in paper
shuffle_mask = False  # shuffles the ETD masks
ft_end = 0  # when to stop ETD training (epochs)
# with no mask; experimental
force_all = False  # forces test to use all neurons without rescaling
train_flip_noise = False  # adds extra noise to training
upscale_mem_during_train = (
    False  # upscale mask by 1/p_mem during training with ETD
)
# Only pick one of the below --
# Otherwise the first one gets picked
drop_no_scaling = True  # uses gen part without rescaling
# and drops rest (original ETD implementation)
p_scale = False  # alternative scaling
tail_comp = (
    1.0  # used with p_scale to account for memorization not being as heavily
)
# utilized as gen part (1.0 has no effect)
rand_dropout_on = False  # dropout of mem that is not in mask
rand_dropout_p = 0.9
