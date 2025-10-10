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

"""Functions for generating and saving plots from experiment results.

This module contains functions to process experiment data, calculate metrics,
and generate various heatmap plots using seaborn and matplotlib, visualizing
results such as healing, utility, and proxy scores. Gets called from the
Experiments_run training loop to create an aggregate small_table.csv and all
heatmap plots. Can be called individually too in case of plot style changes etc.
Prerequisite for tables.py
"""

import copy
import os
import re
import warnings  # for plotting depreciations
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src import config

prefix_folder = config.plots_folder
warnings.filterwarnings("ignore")


def custom_diverging_cmap(vmin, vmax):
  """Creates a custom diverging colormap for seaborn, with specific colors for ranges.

  Args:
      vmin (float): The minimum value for the colormap range.
      vmax (float): The maximum value for the colormap range.

  Returns:
      matplotlib.colors.ListedColormap: The custom colormap.
  """
  colors = [
      "#D10B0C",
      # "#ED3537",
      "#ffffff",
      "#ffffff",
      "#ffffff",
      # "#1494E9",
      "#0D68C3",
  ]
  middle = (vmin + vmax) / 2
  distance = (vmax - middle) * 2
  positions = [
      vmin,
      middle - 0.01 * distance,
      middle,
      middle + 0.01 * distance,
      vmax,
  ]
  # Normalize positions to the range [0, 1]
  norm = mcolors.Normalize(vmin=min(positions), vmax=max(positions))
  normalized_positions = norm(positions)
  # Create the colormap
  cmap = mcolors.LinearSegmentedColormap.from_list(
      "custom_div", list(zip(normalized_positions, colors))
  )
  return cmap


def get_plots(df_annotation, ul_name, tt, rel_, df_primary):
  """Generates and saves various heatmap plots from experiment data.

  This function takes filtered dataframes, pivots them to create heatmaps
  visualizing metrics like healing, utility, and a combined proxy score across
  different discovery rates and unlearning methods. It saves multiple PDF files
  of these plots.

  Args:
      df_annotation (pd.DataFrame): DataFrame containing values used for
        annotations on the heatmaps (e.g., relative differences).
      ul_name (str): The name of the unlearning method being plotted. Used in
        plot titles and filenames.
      tt (str): The training type. Used in plot titles and filenames.
      rel_ (str): A string indicating the type of results ('relative' or
        'absolute'), used to structure the output directory.
      df_primary (pd.DataFrame): DataFrame containing the primary values for the
        heatmaps (e.g., absolute scores).
  """
  # Convert 'discovery rate' to string
  df_primary["discovery rate"] = df_primary["discovery rate"].astype(str)
  df_annotation["discovery rate"] = df_annotation["discovery rate"].astype(str)
  df_primary = df_primary.replace({"poisoning": "Poison Trigger"}, regex=True)
  df_primary = df_primary.replace(
      {"interclasslabelswap": "IC Label Swap"}, regex=True
  )
  df_primary = df_primary.replace(
      {"randomlabelswap": "Rand. Label Swap"}, regex=True
  )
  df_annotation = df_annotation.replace(
      {"poisoning": "Poison Trigger"}, regex=True
  )
  df_annotation = df_annotation.replace(
      {"interclasslabelswap": "IC Label Swap"}, regex=True
  )
  df_annotation = df_annotation.replace(
      {"randomlabelswap": "Rand. Label Swap"}, regex=True
  )
  # Define method order
  method_order = ["Poison Trigger", "IC Label Swap", "Rand. Label Swap"]
  # Create pivot tables
  pivot_fixed_train = df_primary.pivot_table(
      index="dataset_method", columns="discovery rate", values="train_fixed"
  )
  pivot_fixed = df_primary.pivot_table(
      index="dataset_method", columns="discovery rate", values="test_fixed"
  )
  pivot_nomanip = df_primary.pivot_table(
      index="dataset_method", columns="discovery rate", values="test_nomanip"
  )
  pivot_proxy = df_primary.pivot_table(
      index="dataset_method",
      columns="discovery rate",
      values="Utility x Healed",
  )
  pivot_proxy_ulonly = df_primary[
      df_primary["discovery rate"] != "0.0"
  ].pivot_table(
      index="dataset_method",
      columns="discovery rate",
      values="Utility x Healed",
  )
  pivot_fixed_train_rel = df_annotation.pivot_table(
      index="dataset_method", columns="discovery rate", values="train_fixed"
  )
  pivot_fixed_rel = df_annotation.pivot_table(
      index="dataset_method", columns="discovery rate", values="test_fixed"
  )
  pivot_nomanip_rel = df_annotation.pivot_table(
      index="dataset_method", columns="discovery rate", values="test_nomanip"
  )
  pivot_proxy_rel = df_annotation.pivot_table(
      index="dataset_method",
      columns="discovery rate",
      values="Utility x Healed",
  )
  pivot_proxy_ulonly_rel = df_annotation[
      df_annotation["discovery rate"] != "0.0"
  ].pivot_table(
      index="dataset_method",
      columns="discovery rate",
      values="Utility x Healed",
  )
  # Reorder rows
  pivot_fixed_train = pivot_fixed_train.reindex(method_order)
  pivot_fixed = pivot_fixed.reindex(method_order)
  pivot_nomanip = pivot_nomanip.reindex(method_order)
  pivot_proxy = pivot_proxy.reindex(method_order)
  pivot_proxy_ulonly = pivot_proxy_ulonly.reindex(method_order)
  pivot_fixed_train_rel = pivot_fixed_train_rel.reindex(method_order)
  pivot_fixed_rel = pivot_fixed_rel.reindex(method_order)
  pivot_nomanip_rel = pivot_nomanip_rel.reindex(method_order)
  pivot_proxy_rel = pivot_proxy_rel.reindex(method_order)
  pivot_proxy_ulonly_rel = pivot_proxy_ulonly_rel.reindex(method_order)
  # Find the minimum and maximum values across both heatmaps
  vmin = -100  # min(pivot_fixed.min().min(), pivot_nomanip.min().min())
  vmax = 100  # max(pivot_fixed.max().max(), pivot_nomanip.max().max())
  cmap = custom_diverging_cmap(vmin, vmax)
  fig, axes = plt.subplots(
      2,
      2,
      figsize=(15, 8),
      sharey=True,
      gridspec_kw={"width_ratios": [1, 1], "wspace": 0.15, "hspace": 0.2},
  )  # sharey for aligned y-axis and correct plot sizes
  # ---------------------- Healing ----------------------
  sns.heatmap(
      pivot_fixed_train,
      annot=pivot_fixed_train_rel,
      fmt=".1f",
      cmap=cmap,
      ax=axes[0, 0],
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )  # top-left
  axes[0, 0].set_title("Healed (Train)")
  sns.heatmap(
      pivot_fixed,
      annot=pivot_fixed_rel,
      fmt=".1f",
      cmap=cmap,
      ax=axes[0, 1],
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )  # top-right
  axes[0, 1].set_title("Healed (Test)")
  # ---------------------- Proxy ----------------------
  vmin = -100
  vmax = 100
  cmap = custom_diverging_cmap(vmin, vmax)
  sns.heatmap(
      pivot_proxy,
      annot=pivot_proxy_rel,
      fmt=".1f",
      cmap=cmap,
      ax=axes[1, 1],
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )  # bottom-left with shared colorbar
  axes[1, 1].set_title("Utility x Healed")
  # ---------------------- Utility ----------------------
  vmin = 80  # min(pivot_fixed.min().min(), pivot_nomanip.min().min())
  vmax = 100  # max(pivot_fixed.max().max(), pivot_nomanip.max().max())
  cmap_ut = custom_diverging_cmap(vmin, vmax)
  sns.heatmap(
      pivot_nomanip,
      annot=pivot_nomanip_rel,
      fmt=".1f",
      cmap=cmap_ut,
      ax=axes[1, 0],
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )  # bottom-left with shared colorbar
  axes[1, 0].set_title("Utility (Test)")
  # Remove unused subplot
  # fig.delaxes(axes[1,1]) # this can be used to remove plots
  # Hide the y-axis label for the right subplots
  axes[0, 1].set_ylabel("")
  axes[1, 1].set_ylabel("")
  axes[0, 0].set_xlabel("")
  axes[0, 1].set_xlabel("")
  axes[0, 0].axvline(1, color="black", lw=5)
  axes[0, 1].axvline(1, color="black", lw=5)
  axes[1, 0].axvline(1, color="black", lw=5)
  axes[1, 1].axvline(1, color="black", lw=5)
  # do not show the y axis name for left
  axes[0, 0].set_ylabel("Regularity")
  axes[1, 0].set_ylabel("Regularity")
  fig.suptitle(ul_name + " - " + tt, fontsize="x-large")
  plt.tight_layout()
  ul_name = ul_name.replace(".", "")  # remove . to allow saving as file
  if not os.path.exists(f"{prefix_folder}/"):
    os.mkdir(f"{prefix_folder}/")
  if not os.path.exists(f"{prefix_folder}/{rel_}/"):
    os.mkdir(f"{prefix_folder}/{rel_}/")
  if not os.path.exists(f"{prefix_folder}/{rel_}/{tt}/"):
    os.mkdir(f"{prefix_folder}/{rel_}/{tt}/")
  plt.savefig(
      f"{prefix_folder}/{rel_}/{tt}/{tt}_{ul_name}.pdf",
      bbox_inches="tight",
      format="pdf",
  )
  plt.close()
  # -------------------------------
  # just the overview
  vmin = -100
  vmax = 100
  cmap = custom_diverging_cmap(vmin, vmax)
  ax = sns.heatmap(
      pivot_proxy_ulonly,
      annot=pivot_proxy_ulonly_rel,
      fmt=".1f",
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )
  # plt.axvline(1, color='black', lw=5)
  plt.ylabel("")
  path_fig = f"{prefix_folder}_individual/{rel_}/{tt}"
  if not os.path.exists(f"{prefix_folder}_individual/"):
    os.mkdir(f"{prefix_folder}_individual/")
  if not os.path.exists(f"{prefix_folder}_individual/{rel_}/"):
    os.mkdir(f"{prefix_folder}_individual/{rel_}/")
  if not os.path.exists(path_fig):
    os.mkdir(path_fig)
  ax.axvline(0, color="black", lw=4)
  ax.axhline(3, color="black", lw=4)
  plt.savefig(
      path_fig + f"/{tt}_{ul_name}.pdf", bbox_inches="tight", format="pdf"
  )
  plt.close()
  # -------- blank plots
  plt.figure(figsize=(6, 2.5))
  ax = sns.heatmap(
      pivot_proxy_ulonly,
      annot=None,
      fmt=".1f",
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )
  # remove all x,y ticks and labels
  plt.xticks([])
  plt.yticks([])
  plt.xlabel("")
  plt.ylabel("")
  # plt.axvline(1, color='black', lw=5)
  plt.ylabel("")
  path_fig = f"{prefix_folder}_individual/{rel_}blank/{tt}"
  if not os.path.exists(f"{prefix_folder}_individual/"):
    os.mkdir(f"{prefix_folder}_individual/")
  if not os.path.exists(f"{prefix_folder}_individual/{rel_}blank/"):
    os.mkdir(f"{prefix_folder}_individual/{rel_}blank/")
  if not os.path.exists(path_fig):
    os.mkdir(path_fig)
  ax.axvline(0, color="black", lw=4)
  ax.axhline(3, color="black", lw=4)
  plt.savefig(
      path_fig + f"/{tt}_{ul_name}.pdf", bbox_inches="tight", format="pdf"
  )
  plt.close()
  # -------- blank plots
  plt.figure(figsize=(6, 2))
  ax = sns.heatmap(
      pivot_proxy,
      annot=pivot_proxy_rel,
      fmt=".1f",
      cmap=cmap,
      vmin=vmin,
      vmax=vmax,
      cbar=False,
  )
  # plt.axvline(1, color='black', lw=5)
  plt.ylabel("")
  ax.set_yticklabels(["High", "Med.", "Low"])
  path_fig = f"{prefix_folder}_individual/{rel_}all/{tt}"
  if not os.path.exists(f"{prefix_folder}_individual/"):
    os.mkdir(f"{prefix_folder}_individual/")
  if not os.path.exists(f"{prefix_folder}_individual/{rel_}all/"):
    os.mkdir(f"{prefix_folder}_individual/{rel_}all/")
  if not os.path.exists(path_fig):
    os.mkdir(path_fig)
  ax.axvline(0, color="black", lw=4)
  ax.axvline(1, color="black", lw=1, linestyle="--")
  ax.axhline(3, color="black", lw=4)
  plt.savefig(
      path_fig + f"/{tt}_{ul_name}_.pdf", bbox_inches="tight", format="pdf"
  )
  plt.close()
  if ul_name == "NPOPLUS":
    ax = sns.heatmap(
        pivot_fixed_train,
        annot=pivot_fixed_train,
        fmt=".1f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    # plt.axvline(1, color='black', lw=5)
    plt.ylabel("")
    path_fig = f"{prefix_folder}_individual/{rel_}blank/{tt}"
    if not os.path.exists(f"{prefix_folder}_individual/"):
      os.mkdir(f"{prefix_folder}_individual/")
    if not os.path.exists(f"{prefix_folder}_individual/{rel_}blank/"):
      os.mkdir(f"{prefix_folder}_individual/{rel_}blank/")
    if not os.path.exists(path_fig):
      os.mkdir(path_fig)
    ax.axvline(0, color="black", lw=4)
    ax.axvline(1, color="black", lw=1, linestyle="--")
    ax.axhline(3, color="black", lw=4)
    plt.savefig(
        path_fig + f"/{tt}_{ul_name}_trainheal.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()
    ax = sns.heatmap(
        pivot_nomanip,
        annot=pivot_nomanip_rel,
        fmt=".1f",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
    )
    # plt.axvline(1, color='black', lw=5)
    plt.ylabel("")
    path_fig = f"{prefix_folder}_individual/{rel_}blank/{tt}"
    if not os.path.exists(f"{prefix_folder}_individual/"):
      os.mkdir(f"{prefix_folder}_individual/")
    if not os.path.exists(f"{prefix_folder}_individual/{rel_}blank/"):
      os.mkdir(f"{prefix_folder}_individual/{rel_}blank/")
    if not os.path.exists(path_fig):
      os.mkdir(path_fig)
    ax.axvline(0, color="black", lw=4)
    ax.axvline(1, color="black", lw=1, linestyle="--")
    ax.axhline(3, color="black", lw=4)
    plt.savefig(
        path_fig + f"/{tt}_{ul_name}_utility.pdf",
        bbox_inches="tight",
        format="pdf",
    )
    plt.close()


def subtract_baseline(input_df_filter, subtract_col):
  """Subtracts a baseline value from a specified column in the DataFrame.

  For each unique 'dataset_method', this function identifies the row with
  'UL_type' == "Baseline". The value in `subtract_col` from this baseline row is
  then subtracted from all other rows within the same 'dataset_method' group in
  the `subtract_col`.

  Args:
      input_df_filter (pd.DataFrame): The input DataFrame containing experiment
        data.
      subtract_col (str): The name of the column from which the baseline value
        should be subtracted.

  Returns:
      pd.DataFrame: A new DataFrame with the baseline subtracted from the
      `subtract_col`. Returns the original DataFrame if an error occurs.
  """
  try:
    df_copy = input_df_filter.copy()
    for task in df_copy["dataset_method"].unique():
      task_filter = df_copy["dataset_method"] == task
      baseline_filter = (df_copy["UL_type"] == "Baseline") & task_filter
      non_baseline_filter = (
          task_filter  # (df_copy["UL_type"] != "Baseline") & task_filter
      )
      # print(baseline_filter)
      # print(df_copy[df_copy["UL_type"] == "Baseline"])
      if (
          baseline_filter.sum() == 1
      ):  # check that we have only one baseline per task
        baseline_value = input_df_filter.loc[
            baseline_filter, subtract_col
        ].iloc[0]
        # update the rows that are not Baseline in test_nomanip
        df_copy.loc[non_baseline_filter, subtract_col] = (
            df_copy.loc[non_baseline_filter, subtract_col] - baseline_value
        )
        # print("SUCCESS")
      elif baseline_filter.sum() > 1:
        print(
            f"Warning: More than one Baseline found for task {task}. Skipping"
            " subtraction for this task."
        )
        print(df_copy[df_copy["UL_type"] == "Baseline"])
      else:
        print(
            f"Warning: No Baseline found for task {task}. Skipping subtraction"
            " for this task."
        )
    return df_copy
  except (KeyError, TypeError, IndexError) as e:
    print(f"Error during operation: {e}")
    print("Returning original DataFrame.")
    return input_df_filter


if __name__ == "__main__":
  ####################################################################################
  best_column = "select_proxy"  # to pick from all the runs
  detail_type = False
  manual_stitching = False
  if manual_stitching:  # for manual combos
    df1 = pd.read_table("small_table_seed0.csv", sep=",")
    df2 = pd.read_table("small_table_seed1.csv", sep=",")
    df3 = pd.read_table("small_table_seed2.csv", sep=",")
    df = pd.concat([df1, df2, df3])
    prefix_folder = "CF"
  else:
    csv_table = pd.read_table("results.csv", sep=",")
    try:  # if randomlabelswap is included
      csv_table = csv_table[[
          "train_type",
          "dataset",
          "model",
          "dataset_method",
          "forget_set_size",
          "pretrain_iters",
          "unlearn_method",
          "exp_name",
          "alpha",
          "test_acc",
          "test_clean_acc",
          "deletion_size",
          "unlearn_time",
          "manip_acc",
          "train_clean_acc",
          "test_retain_acc",
          "manip_clean_acc",
          "delete_acc",
          "delete_err",
          "forget_acc",
          "run_name",
      ]]
      for index, row in csv_table.iterrows():
        if row["dataset_method"] == "randomlabelswap":
          csv_table.loc[index, "manip_acc"] = csv_table.loc[
              index, "forget_acc"
          ]  # randomlabelswap has differnt logging...
          # fixing it here for consistent plots
          csv_table.loc[index, "test_retain_acc"] = csv_table.loc[
              index, "test_acc"
          ]
    except KeyError:
      csv_table = csv_table[[
          "train_type",
          "dataset",
          "model",
          "dataset_method",
          "unlearn_time",
          "forget_set_size",
          "pretrain_iters",
          "unlearn_method",
          "exp_name",
          "alpha",
          "test_acc",
          "test_clean_acc",
          "deletion_size",
          "manip_acc",
          "train_clean_acc",
          "test_retain_acc",
          "manip_clean_acc",
          "delete_acc",
          "delete_err",
          "run_name",
      ]]
    # preprocessing
    csv_table["discovery"] = (
        csv_table["deletion_size"] / csv_table["forget_set_size"]
    )
    # change train_type to EU when unlearn_method == "EU"
    # csv_table.loc[csv_table['unlearn_method'] == "EU", 'train_type'] = "EU"
    # helper columns
    csv_table["avg_score"] = np.where(
        csv_table["dataset_method"] == "poisoning",
        (csv_table["test_acc"] + csv_table["test_clean_acc"]) / 2,
        (csv_table["test_acc"] + csv_table["test_retain_acc"]) / 2,
    )
    # easy check for methods used for easier access
    methods_to_check = ["Naive", "LLF", "ETD", "EU"]
    csv_table["pretrain"] = ""
    for checking in methods_to_check:
      csv_table.loc[
          csv_table["train_type"].str.contains(checking, case=False, na=False),
          "pretrain",
      ] += checking  # += for combinations of methods
    # create a new simplified table
    small_table = csv_table[[
        "train_type",
        "unlearn_method",
        "dataset",
        "dataset_method",
        "forget_set_size",
        "deletion_size",
        "alpha",
        "model",
        "manip_acc",
        "train_clean_acc",
        "delete_acc",
        "test_acc",
        "test_clean_acc",
        "test_retain_acc",
        "run_name",
        "unlearn_time",
    ]]
    # create one column for retain across tasks
    small_table["test_clean_acc"] = small_table["test_clean_acc"].fillna(0)
    small_table["test_retain_acc"] = small_table["test_retain_acc"].fillna(0)
    small_table["test_retain"] = (
        small_table["test_clean_acc"] + small_table["test_retain_acc"]
    )
    small_table = small_table.drop(
        ["test_retain_acc", "test_clean_acc"], axis=1
    )
    small_table = small_table.sort_values([
        "dataset_method",
        "dataset",
        "forget_set_size",
        "deletion_size",
        "unlearn_method",
        "train_type",
        "alpha",
    ])
    small_table = small_table.drop_duplicates()
    small_table = small_table.rename(
        columns={
            "test_acc": "test_fixed",
            "forget_set_size": "Df_size",
            "unlearn_method": "UL_type",
            "manip_acc": "train_fixed",
            "deletion_size": "del_size",
            "delete_acc": "train_forget_fixed",
            "train_clean_acc": "train_nomanip",
            "test_retain": "test_nomanip",
            "unlearn_time": "time",
        },
        errors="raise",
    )
    # ensure no nan
    small_table["test_nomanip"] = small_table["test_nomanip"].fillna(0)
    small_table["test_fixed"] = small_table["test_fixed"].fillna(0)
    small_table["train_fixed"] = small_table["train_fixed"].fillna(0)
    small_table["train_nomanip"] = small_table["train_nomanip"].fillna(0)
    # calc proxy scores
    small_table["avg_test"] = (
        small_table["test_fixed"] + small_table["test_nomanip"]
    ) / 2
    small_table["avg_train"] = (
        small_table["train_fixed"] + small_table["train_nomanip"]
    ) / 2
    small_table["IC_proxy"] = (
        small_table["train_fixed"] + small_table["test_nomanip"]
    ) / 2
    small_table["select_proxy"] = (
        small_table["train_fixed"] * small_table["test_nomanip"]
    )
    # ranking cols
    small_table["best_test_fixed"] = ""
    small_table["best_test_nomanip"] = ""
    # SCRUB comparison at alpha level
    scrub_rows = small_table["UL_type"] == "Scrub"
    small_table.loc[scrub_rows, "UL_type"] = "Scrub_" + small_table.loc[
        scrub_rows, "alpha"
    ].astype(str)

    def mark_top_performers(input_df):
      """Marks the top three performers within each group with asterisks."""

      def mark_top(group, column_name, marker="o"):
        """Marks top performers within a group."""
        ranked = group.nlargest(3, column_name)
        group[f"best_{column_name}"] = ""  # Initialize the new column
        for i, index_i in enumerate(ranked.index):
          if i == 0:
            group.loc[index_i, f"best_{column_name}"] = marker * 3
          elif i == 1:
            group.loc[index_i, f"best_{column_name}"] = marker * 2
          elif i == 2:
            group.loc[index_i, f"best_{column_name}"] = marker
        return group

      # Group by the specified columns and apply the marking function
      grouped = input_df.groupby(
          ["dataset", "dataset_method", "Df_size", "del_size"]
      )
      processed_df = grouped.apply(
          mark_top, column_name="test_fixed"
      ).reset_index(drop=True)
      grouped = processed_df.groupby(
          ["dataset", "dataset_method", "Df_size", "del_size"]
      )
      processed_df = grouped.apply(
          mark_top, column_name="test_nomanip"
      ).reset_index(drop=True)
      grouped = processed_df.groupby(
          ["dataset", "dataset_method", "Df_size", "del_size"]
      )
      processed_df = grouped.apply(
          mark_top, column_name="avg_test", marker="X"
      ).reset_index(drop=True)
      grouped = processed_df.groupby(
          ["dataset", "dataset_method", "Df_size", "del_size"]
      )
      processed_df = grouped.apply(
          mark_top, column_name="avg_train", marker="T"
      ).reset_index(drop=True)
      return processed_df

    def get_best_scrub_and_non_scrub(input_df):
      """Selects the 'Scrub' row with the highest average of 'test_fixed' and 'test_nomanip'.

      AND all non-'Scrub' rows for each unique combination of dataset,
      dataset_method, Df_size, and del_size.

      Args:
          input_df (pd.DataFrame): The input DataFrame containing experiment
            results.

      Returns:
          pd.DataFrame: A DataFrame containing the best 'Scrub' row and all
            non-'Scrub' rows for each group.
      """

      def get_best_row(group):
        """Selects the best 'Scrub' row within a group and all non-Scrub rows."""
        scrub_group = group[group["UL_type"] == "Scrub"]
        non_scrub_group = group[group["UL_type"] != "Scrub"]
        best_rows = non_scrub_group.copy()  # Add all non scrub rows
        if not scrub_group.empty:
          best_scrub_row = scrub_group.nlargest(1, "select_proxy")  # avg_test
          best_rows = pd.concat(
              [best_rows, best_scrub_row]
          )  # Add best scrub row
        return best_rows

      grouped = input_df.groupby([
          "dataset",
          "dataset_method",
          "Df_size",
          "del_size",
          "UL_type",
          "train_type",
      ])
      best_df = grouped.apply(get_best_row).reset_index(drop=True)
      return best_df

    def simplify_train_type(df_input, not_simple=False):
      """Simplifies the 'train_type' column based on the given rules."""
      df_input["train_type_simplified"] = ""  # Initialize the new column
      if not_simple:
        for index_i, row_i in df_input.iterrows():
          val_simple = row_i["train_type"]
          simple_name = f"{val_simple}"
          df_input.loc[index_i, "train_type_simplified"] = simple_name
      else:
        for index_i, row_i in df_input.iterrows():
          simple_name = ""
          if "Naive" in row_i["train_type"]:
            simple_name += "Baseline "
          if "drop0" in row_i["train_type"]:
            simple_name += "ETD no drop "
          if "drop1" in row_i["train_type"]:
            simple_name += "ETD drop "
          if "drop2" in row_i["train_type"]:
            simple_name += "ETD IDEAL drop (all) "
          if "drop3" in row_i["train_type"]:
            simple_name += "ETD IDEAL no drop "
          if "drop4" in row_i["train_type"]:
            simple_name += "ETD IDEAL drop (Df) "
          if "LLF" in row_i["train_type"]:
            simple_name += "LLF "
          if "LOSSDROP" in row_i["train_type"]:
            simple_name += "Trunc. loss "
          if not simple_name:
            val_simple = row_i["train_type"]
            simple_name = f"Not yet listed: {val_simple}"
          df_input.loc[index_i, "train_type_simplified"] = simple_name
      return df_input

    def get_best_for_each(
        input_df,
    ):
      """Selects the best performing row for each unique UL_type and train_type_simplified.

      Within each combination of 'UL_type' and 'train_type_simplified', this
      function finds the row with the highest value in the column specified by
      `best_column` and returns a DataFrame containing only these best rows
      along with all other rows that were not part of these specific groups.

      Args:
          input_df (pd.DataFrame): The input DataFrame containing experiment
            results.

      Returns:
          pd.DataFrame: A DataFrame with only the best row selected for each
            'UL_type' and 'train_type_simplified' group, based on `best_column`.
      """

      # Note scrub is just a placeholder for method
      def get_best_row_each(group):
        unq_types = group["UL_type"].unique()
        print(unq_types)
        for unq in unq_types:
          sub_group = group[group["UL_type"] == unq]
          unq_pre = sub_group["train_type_simplified"].unique()
          for unqpre in unq_pre:
            scrub_group = group.loc[
                (group["train_type_simplified"] == unqpre)
                & (group["UL_type"] == unq)
            ]
            non_scrub_group = group.loc[
                ~(
                    (group["train_type_simplified"] == unqpre)
                    & (group["UL_type"] == unq)
                )
            ]
            best_rows = non_scrub_group.copy()
            if not scrub_group.empty:
              best_scrub_row = scrub_group.nlargest(1, best_column)
              group = pd.concat([best_rows, best_scrub_row])
        return group

      best_df = get_best_row_each(input_df)
      return best_df

    # small_table = get_best_scrub_and_non_scrub(small_table)
    # Filter if necessary to only show one del size etc.
    # small_table = small_table[small_table.dataset != "CIFAR100"]
    # small_table = small_table[small_table.dataset == "CIFAR10"]
    # small_table = small_table[small_table.del_size == 1000]
    # small_table = small_table[small_table.Df_size == 1000]
    small_table = mark_top_performers(small_table)
    small_table = small_table.rename(
        columns={
            "best_test_fixed": "best_fix",
            "best_test_nomanip": "best_ret",
        },
        errors="raise",
    )
    small_table = simplify_train_type(small_table, not_simple=detail_type)
    small_table["exp"] = (
        small_table["dataset"]
        + " "
        + small_table["dataset_method"]
        + ": "
        + small_table["Df_size"].astype(str)
        + " samples ("
        + small_table["del_size"].astype(str)
        + " discovered)"
    )

    # relative improvements to starting point
    # for each row find the row that has the same values for train_type,
    # dataset, dataset_method, Df_size and has UL_type=="Naive" then take the
    # difference between these two columns for each of 'train_fixed',
    # 'train_nomanip', 'train_forget_fixed', 'test_fixed', 'test_nomanip',
    # 'avg_test', and save them as new cols (e.g., rel_train_fixed, rel_...)
    def calculate_relative_differences(my_table):
      """Calculates relative differences compared to 'Naive' UL_type."""
      relative_cols = [
          "train_fixed",
          "train_nomanip",
          "train_forget_fixed",
          "test_fixed",
          "test_nomanip",
          "avg_test",
          "avg_train",
          "IC_proxy",
      ]
      relative_prefix = "rel_"
      # Set the MultiIndex
      my_table = my_table.set_index(
          ["train_type", "dataset", "dataset_method", "Df_size"]
      )
      # Get the "Naive" rows,
      # selecting the one with highest 'avg' if multiple exist
      naive_rows = (
          my_table[my_table["UL_type"] == "Naive"]
          .sort_values("avg_train")
          .groupby(level=[0, 1, 2, 3])
          .last()
      )
      # Reset the index of naive_rows so we can merge on columns
      naive_rows = naive_rows.reset_index()
      my_table = my_table.reset_index()
      # Use a LEFT merge to keep all rows from the original table
      merged_table = my_table.merge(
          naive_rows,
          on=["train_type", "dataset", "dataset_method", "Df_size"],
          how="left",
          suffixes=("", "_Naive"),
      )
      for col in relative_cols:
        new_col_name = relative_prefix + col
        merged_table[new_col_name] = (
            merged_table[col] - merged_table[col + "_Naive"]
        )
      merged_table["Compared rows"] = merged_table["UL_type"] + merged_table[
          "UL_type_Naive"
      ].fillna("")
      merged_table = merged_table.drop(
          columns=[col + "_Naive" for col in ["UL_type"] + relative_cols],
          errors="ignore",
      )
      return merged_table

    small_table = calculate_relative_differences(small_table)
    # extract params for easy access
    to_extract = ["LLFe", "l", "fix", "mem", "drop"]
    for index, row in small_table.iterrows():
      # small_table[extract_val] = np.nan
      for extract_val in to_extract:
        pattern = r"{extract_val}(\d+)".format(
            extract_val=extract_val
        )  # String formatting
        match_param = re.search(pattern, row["train_type"])
        # match_param = re.search(r"{extract_val}(\d+)", row["train_type"])
        # print(extract_val, match_param, row["train_type"], pattern)
        if match_param is not None:
          param_val = int(match_param.group(1))
          small_table.loc[index, extract_val] = param_val
    small_table = small_table.round(3)
    small_table = small_table.replace({"Naive": "Starting Point"}, regex=True)
    small_table["train_type"] = small_table["train_type"].replace(
        {"Starting Point": "Baseline"}, regex=True
    )
    small_table = small_table.replace({"EU": "Retrained"}, regex=True)
    small_table = small_table.replace({"XALFSSD": "Potion"}, regex=True)
    # renaming for multiple runs (hacky way)
    print("small_table.columns", small_table.columns)
    small_table = small_table.replace(
        {"RAYNnomem": "RAYN (g. only)"}, regex=True
    )  # renaming of an old acronym we used for a REM version
    # csv_table.head()
    # print(small_table.columns)
    small_table = small_table.fillna(0)
    small_table.head(5)
    # drop rows where run_name==0
    small_table = small_table[small_table["run_name"] != 0]
    print(small_table.shape)
    print(small_table.shape)
    print("===================")
    small_table.to_csv("small_table.csv", index=False, sep=",")
    # Create folder if does not exist
    if not os.path.exists(f"{prefix_folder}/"):
      os.makedirs(f"{prefix_folder}/")
    small_table.to_csv(
        f"{prefix_folder}/runs.csv", index=False, sep=","
    )  # all in one folder
    small_table["run_name"].unique()
    small_table["UL_type"].unique()
    df = small_table
  ###########################################
  # df =pd.read_table("small_table.csv",sep=',')
  df["discovery rate"] = df["del_size"] / df["Df_size"]
  df["discovery rate"] = np.round(df["discovery rate"], 5)
  df["train_fixed"] *= 100
  df["test_nomanip"] *= 100
  df["test_fixed"] *= 100
  df["Utility x Healed"] = df["train_fixed"] * df["test_nomanip"] / 100
  # Renaming of Methods
  df["UL_type"] = df["UL_type"].replace({"NEW": "NEW"}, regex=True)
  df["UL_type"] = df["UL_type"].replace({"BadT": "Bad Teacher"}, regex=True)
  df["UL_type"] = df["UL_type"].replace({"CF": "Finetuning"}, regex=True)
  df["UL_type"] = df["UL_type"].replace({"Scrub": "SCRUB"}, regex=True)
  df["UL_type"] = df["UL_type"].replace(
      {"Starting Point": "Baseline"}, regex=True
  )
  # rename training
  df["train_type"] = df["train_type"].replace({"ETDfix": "Gen "}, regex=True)
  df["train_type"] = df["train_type"].replace({"mem": " Mem "}, regex=True)
  df["train_type"] = df["train_type"].replace({"drop1": " (drop)"}, regex=True)
  df["train_type"] = df["train_type"].replace(
      {"drop0": " (no drop)"}, regex=True
  )
  # --- Add new rows for Baseline with different discovery rates ---
  baseline_rows = df[df["UL_type"] == "Baseline"].copy()
  new_rows = []
  discovery_rates = [
      "0.1",
      "0.2",
      "0.3",
      "0.4",
      "0.5",
      "0.6",
      "0.7",
      "0.8",
      "0.9",
      "1.0",
  ]
  for _, row in baseline_rows.iterrows():
    for rate in discovery_rates:
      new_row = row.copy()
      new_row["discovery rate"] = rate
      new_row.loc["UL_type"] = "Trained"
      new_rows.append(new_row)
  df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
  for relative in [
      False
  ]:  # add true here to get relative results too (if of interest)
    if relative:
      rel = "relative"
    else:
      rel = "absolute"
    rel_base = rel
    for model in df["model"].unique():
      for dataset in df[df["model"] == model]["dataset"].unique():
        df_scenario = df[df["model"] == model]
        df_scenario = df_scenario[df_scenario["dataset"] == dataset]
        print(
            "====> Plotting, ",
            model,
            dataset,
            df_scenario["del_size"].unique(),
            df_scenario.shape,
        )
        for train_type in df["train_type"].unique():
          if os.path.exists(f"{prefix_folder}/{rel}/{train_type}"):
            pass
            # shutil.rmtree(f"{prefix_folder}/{rel}/{train_type}")
          for ul in df["UL_type"].unique():
            df_filter = copy.deepcopy(
                df_scenario
            )  # otherwise we filter to empty after an iteration
            rel = rel_base + "_" + model + "_" + dataset
            # df_filter = df[df["run_name_Naive"]=="inflation_01maxperc"]
            df_filter = df_filter.loc[
                :,
                [
                    "dataset",
                    "del_size",
                    "test_fixed",
                    "train_fixed",
                    "test_nomanip",
                    "Utility x Healed",
                    "discovery rate",
                    "dataset_method",
                    "UL_type",
                    "train_type",
                    "run_name_Naive",
                ],
            ]  # "run_name_Naive"
            df_filter = df_filter[df_filter["train_type"] == train_type]
            df_filter = df_filter[
                (df_filter["UL_type"] == ul)
                | (df_filter["UL_type"] == "Baseline")
            ]
            print("======>", ul, df_filter["del_size"].unique())
            df_filter_rel = copy.deepcopy(df_filter)
            if relative:
              df_filter = subtract_baseline(df_filter, "test_nomanip")
              df_filter = subtract_baseline(df_filter, "test_fixed")
              df_filter = subtract_baseline(df_filter, "train_fixed")
              df_filter = subtract_baseline(df_filter, "Utility x Healed")
              df_filter_rel = subtract_baseline(df_filter_rel, "test_nomanip")
              df_filter_rel = subtract_baseline(df_filter_rel, "test_fixed")
              df_filter_rel = subtract_baseline(df_filter_rel, "train_fixed")
              df_filter_rel = subtract_baseline(
                  df_filter_rel, "Utility x Healed"
              )
            get_plots(df_filter_rel, ul, train_type, rel, df_filter)
