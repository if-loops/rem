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

"""Creates LateX ready tables.

Prerequisite to run plotting.py Both are automatically executed at the end of
training with Experiments_run.sh
"""

import os
import numpy as np
import pandas as pd


def pandas_to_latex_combined_mean_std_highlight(dataframe, caption):
  """Converts a pandas DataFrame with mean/sem columns to a LaTeX table.

  This function takes a pandas DataFrame, typically the result of an aggregation
  with 'mean' and 'sem' columns for different metrics. It formats these into a
  LaTeX table, including highlighting: - Bold text for the best mean value in
  each metric column. - Blue text for the second best mean value in each metric
  column. The function also performs several string replacements on the
  DataFrame values to clean up labels for the table.

  Args:
    dataframe: pandas.DataFrame. Expected to have a MultiIndex in columns, with
      at least 'mean' and 'sem' under various metric names.
    caption: str. The caption to be used for the LaTeX table.

  Returns:
    str: A string containing the complete LaTeX table environment.
  """
  processed_df = dataframe.sort_values(
      by=('Utility x Healed', 'mean'), ascending=False
  )
  processed_df = processed_df.replace({'ETDfix': 'Capacity 0.'}, regex=True)
  processed_df = processed_df.replace({'mem20drop1': ' (ETD)'}, regex=True)
  processed_df = processed_df.replace({'mem0drop1': ''}, regex=True)
  processed_df = processed_df.replace(
      {'mem20drop2': ' (ETD IDEAL)'}, regex=True
  )
  processed_df = processed_df.replace({'Baseline': 'Capacity 1.0'}, regex=True)
  processed_df = processed_df.replace({'REM': 'REM'}, regex=True)
  processed_df = processed_df.replace({'REMIDEAL': 'REM (IDEAL)'}, regex=True)
  # Identify grouping columns (single level index)
  # and metric columns (multi-level index)
  grouping_cols = [
      col[0]
      for col in processed_df.columns
      if isinstance(col, tuple) and len(col) > 1 and not col[1]
  ]
  # Ensure metric_cols are unique and sorted for consistent column order
  metric_cols = sorted(
      list(
          set(
              col[0]
              for col in processed_df.columns
              if isinstance(col, tuple)
              and len(col) > 1
              and col[1] in ['mean', 'sem']
          )
      )
  )
  # --- Pre-calculate best and second best means for each metric column ---
  best_values = {}
  for metric in metric_cols:
    mean_col = (metric, 'mean')
    if mean_col in processed_df.columns:
      # Extract numeric mean values, drop NaNs for comparison
      means = pd.to_numeric(processed_df[mean_col], errors='coerce').dropna()
      if not means.empty:
        # Find the top 2 unique values
        unique_sorted_means = sorted(means.unique(), reverse=True)
        best_mean = unique_sorted_means[0]
        second_best_mean = (
            unique_sorted_means[1] if len(unique_sorted_means) > 1 else None
        )
        best_values[metric] = (best_mean, second_best_mean)
      else:
        best_values[metric] = (None, None)  # No valid means found
    else:
      best_values[metric] = (None, None)  # Mean column doesn't exist
  # --- Start LaTeX table string ---
  latex_str = ''
  latex_str += '\\begin{table}\n\\centering\n'
  latex_str += (
      f"\\begin{{tabular}}{{{'l' * len(grouping_cols)}{'c' * len(metric_cols)}}}\n"
  )
  latex_str += '\\toprule\n'
  # --- Write header row ---
  # Clean up headers (replace underscores, title case)
  grouping_headers = [str(col).replace('_', ' ') for col in grouping_cols]
  metric_headers = [str(col).replace('_', ' ') for col in metric_cols]
  all_headers = grouping_headers + metric_headers
  latex_str += ' & '.join(all_headers) + ' \\\\\n'
  latex_str += '\\midrule\n'
  # --- Write data rows ---
  for _, row in processed_df.iterrows():
    row_values = []
    # Add grouping column values first
    for group_col in grouping_cols:
      # Handle potential MultiIndex for row index name if needed,
      # otherwise use simple index
      # For grouping cols, access directly
      row_values.append(str(row[(group_col, '')]))
    # Add metric column values (mean ± std) with highlighting
    for metric in metric_cols:
      mean_col = (metric, 'mean')
      std_col = (metric, 'sem')
      # Check if mean and std columns exist for this metric
      if mean_col in row and std_col in row:
        mean_val = row[mean_col]
        std_val = row[std_col]
        # Format as "mean \pm std"
        # Use a consistent format specifier (e.g., .2f)
        # Handle potential non-numeric values gracefully before formatting
        try:
          # Try converting to float for comparison and formatting
          mean_val_num = float(mean_val)
          std_val_num = float(std_val)
          base_str = f'{mean_val_num:.2f} $\\pm$ {std_val_num:.2f}'
        except (ValueError, TypeError):
          # If conversion fails, just use original string representation
          base_str = f'{mean_val} $\\pm$ {std_val}'
          mean_val_num = None  # Cannot compare if not numeric
        # Apply highlighting based on pre-calculated best/second best
        combined_val = base_str  # Default: no highlighting
        if metric in best_values and mean_val_num is not None:
          best_mean, second_best_mean = best_values[metric]
          # Use numpy.isclose for robust float comparison if needed,
          if best_mean is not None and np.isclose(mean_val_num, best_mean):
            combined_val = f'\\textbf{{{base_str}}}'
          elif second_best_mean is not None and np.isclose(
              mean_val_num, second_best_mean
          ):
            combined_val = f'\\textcolor{{blue}}{{{base_str}}}'
        row_values.append(combined_val)
      else:
        row_values.append('-')  # Placeholder if mean/std missing for a metric
    latex_str += ' & '.join(row_values) + ' \\\\\n'
  # --- End LaTeX table ---
  latex_str += '\\bottomrule\n'
  latex_str += '\\end{tabular}\n'
  latex_str += f'\\caption{{{caption}}}\n'
  latex_str += f'\\label{{lab:{caption}}}\n'
  latex_str += '\\end{table}\n'
  return latex_str


if __name__ == '__main__':
  df = pd.read_table('small_table.csv', sep=',')
  print(df.columns)
  df['score'] = df['test_nomanip'] * df['train_fixed']
  # PICK desired SCRUB version by renaming
  # to SCRUB from the alpha indexed version
  df = df.replace({'Scrub_0.01': 'SCRUB'}, regex=True)
  # easier usage
  df['Utility x Healed'] = df['test_nomanip'] * df['train_fixed']
  df['Utility'] = df['test_nomanip']
  df['Healed'] = df['train_fixed']
  df['Utility x Healed'] *= 100
  df['Utility'] *= 100
  df['Healed'] *= 100
  # Options
  folder_prefix = ''
  UL_type = [
      'Starting Point',
      'Ascent',
      'BadT',
      'CF',
      'Retrained',
      'NPO',
      'REM',
      'REMGEN',
      'REMIDEAL',
      'NPORAYN',
      'RAYN',
      'SalUn',
      'SCRUB',
      'TrashCollector',
      'Potion',
  ]
  dataset = ['Imagenette']
  train_type = df['train_type'].unique()
  folder_name_tables = folder_prefix + 'results_tables'
  if not os.path.exists(f'{folder_name_tables}/'):
    os.mkdir(f'{folder_name_tables}/')
    print(f'{folder_name_tables}/')
  # iterate over all existing train type, dataset,
  # model combinations for the basic tables
  # (manual combos of multiple can be done if wanted with similar code)
  required_columns = ['model', 'dataset', 'Df_size', 'train_type']
  unique_combinations = df[required_columns].drop_duplicates()
  print(f'Found {len(unique_combinations)} unique combinations.\n')
  # Loop through each unique combination
  for index, combo_row in unique_combinations.iterrows():
    # Create a string of the unique values for the current combination
    # This uses an f-string to format the output clearly
    combo_string = (
        f"Model: {combo_row['model']}, "
        f"Dataset: {combo_row['dataset']}, "
        f"DF Size: {combo_row['Df_size']}, "
        f"Train Type: {combo_row['train_type']}"
    )
    print(f'--- Processing Combination: {combo_string} ---')
    # Filter the original DataFrame for the current combination
    # We build a boolean mask by combining conditions
    # for each column in the combination
    df_selected = df[
        (df['model'] == combo_row['model'])
        & (df['dataset'] == combo_row['dataset'])
        & (df['Df_size'] == combo_row['Df_size'])
        & (df['train_type'] == combo_row['train_type'])
    ].copy()
    df_agg = df_selected[
        df_selected['UL_type'].isin([
            'Starting Point',
            'Ascent',
            'BadT',
            'CF',
            'Retrained',
            'NPO',
            'REM',
            'SCRUB',
            'Potion',
            'SalUn',
        ])
    ]
    df_agg = (
        df_agg.groupby(['UL_type'])
        .agg({
            'Healed': ['mean', 'sem'],
            'Utility': ['mean', 'sem'],
            'Utility x Healed': ['mean', 'sem'],
        })
        .round(2)
        .reset_index()
    )
    latex_table_string = pandas_to_latex_combined_mean_std_highlight(
        df_agg, {combo_string}
    )
    with open(f'{folder_name_tables}/{combo_string}.txt', 'w') as text_file:
      text_file.write(latex_table_string)
