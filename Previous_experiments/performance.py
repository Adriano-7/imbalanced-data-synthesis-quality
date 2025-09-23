#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader


sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'figure.dpi': 100,
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})


# In[2]:


DATA_DIR = "Imbalanced_Datasets/Hepatitis_C_Virus/"
BALANCED_DATA_PATH = os.path.join(DATA_DIR, "hcv_dataset_ir_1.csv")
TARGET_COLUMN = "Fibrosis_Stage_Label"
RANDOM_STATE = 42
TEST_SET_SIZE = 0.3

GENERATIVE_MODELS = ["ctgan", "tvae", "rtvae"]


# In[3]:


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies label encoding to all object/categorical columns in the dataframe,
    ensuring it's ready for machine learning models.
    """
    df_processed = df.copy()

    for col in df_processed.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

    return df_processed

def plot_results(df: pd.DataFrame, metrics: list, title: str, filename: str):
    """
    Generates and saves line plots for specified metrics vs. Imbalance Ratio.
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 6 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes] 

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.lineplot(
            data=df,
            x='IR',
            y=metric,
            hue='Model',
            style='Model',
            marker='o',
            markersize=8,
            ax=ax
        )
        ax.set_title(f'Analysis of {metric}')
        ax.set_ylabel(metric)
        ax.set_xlabel('Imbalance Ratio (IR) - Log Scale')
        ax.set_xscale('log')
        ax.set_xticks(df['IR'].unique())
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both", ls="--")
        ax.legend(title='Generative Model')

    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to '{filename}'")
    plt.show()


print(" Starting Synthetic Data Quality Benchmark ")

print(f"Loading balanced dataset from: {BALANCED_DATA_PATH}")
df_balanced = pd.read_csv(BALANCED_DATA_PATH)
df_processed_balanced = preprocess_data(df_balanced)

# Create a stratified hold-out test set to ensure fair and unbiased evaluation
X = df_processed_balanced.drop(columns=[TARGET_COLUMN])
y = df_processed_balanced[TARGET_COLUMN]
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SET_SIZE, random_state=RANDOM_STATE, stratify=y
)
df_test = pd.concat([X_test, y_test], axis=1)
test_loader = GenericDataLoader(df_test, target_column=TARGET_COLUMN)
print(f"Hold-out test set created. Shape: {df_test.shape}")
print("Test set class distribution:\n", df_test[TARGET_COLUMN].value_counts().sort_index())
print("-" * 50)

imbalanced_files = glob.glob(os.path.join(DATA_DIR, "hcv_dataset_ir_*.csv"))
imbalanced_datasets = {}

for file_path in sorted(imbalanced_files, key=lambda x: int(re.search(r'ir_(\d+)', x).group(1))):
    if (ir_match := re.search(r'ir_(\d+)', file_path)):
        ir = int(ir_match.group(1))
        print(f"Loading training set for IR={ir}...")
        df_train = pd.read_csv(file_path)
        imbalanced_datasets[ir] = preprocess_data(df_train)

print(f"\nLoaded {len(imbalanced_datasets)} training datasets with IRs: {list(imbalanced_datasets.keys())}")
print("-" * 50)



# In[ ]:


results_list = []

for ir, train_df in imbalanced_datasets.items():
    print(f"\n{'='*25} Benchmarking for Imbalance Ratio (IR) = {ir} {'='*25}")
    print(f"Training data shape: {train_df.shape}")

    train_loader = GenericDataLoader(train_df, target_column=TARGET_COLUMN)
    evaluation_tasks = [(f"{model}_ir{ir}", model, {}) for model in GENERATIVE_MODELS]

    metrics_suite = {
        "fidelity": ["alpha_precision"], # Measures how well synthetic data captures real data distribution
        "utility": ["detection_xgb"],  # Measures downstream performance (AUROC, F1) using XGBoost
        "privacy": ["dcr_recap"],      # Distance to Closest Record
    }

    try:
        score = Benchmarks.evaluate(
            tests=evaluation_tasks,
            X=train_loader,
            X_test=test_loader,
            synthetic_size=len(train_df),
            metrics=metrics_suite,
            task_type="classification",
            repeats=1,  # Increase for more robust statistical measures
            device="cuda",
            random_state=RANDOM_STATE,
            workspace="." # temp dir for synthcity
        )

        for model in GENERATIVE_MODELS:
            test_name = f"{model}_ir{ir}"
            results_list.append({
                "IR": ir,
                "Model": model.upper(),
                "Alpha Precision (Fidelity)": score[test_name].loc['fidelity.alpha_precision']['mean'],
                "AUROC (Utility)": score[test_name].loc['utility.detection_xgb.aucroc']['mean'],
                "F1 Macro (Utility)": score[test_name].loc['utility.detection_xgb.f1_score_macro']['mean'],
                "DCR (Privacy)": score[test_name].loc['privacy.dcr_recap']['mean']
            })

    except Exception as e:
        print(f"ERROR during benchmarking for IR={ir}: {e}. Skipping this run.")


# In[ ]:


results_df = pd.DataFrame(results_list)

print("\n\nBenchmark Complete. Final Results:")
print(results_df.round(4).to_string())

results_df.to_csv("hcv_imbalance_benchmark_results.csv", index=False)
print("\nResults saved to 'hcv_imbalance_benchmark_results.csv'")

plot_results(
    results_df,
    metrics=["AUROC (Utility)", "F1 Macro (Utility)"],
    title="Model Utility vs. Imbalance Ratio",
    filename="model_utility_evaluation.png"
)
plot_results(
    results_df,
    metrics=["Alpha Precision (Fidelity)"],
    title="Data Fidelity vs. Imbalance Ratio",
    filename="data_fidelity_evaluation.png"
)
plot_results(
    results_df,
    metrics=["DCR (Privacy)"],
    title="Data Privacy vs. Imbalance Ratio",
    filename="data_privacy_evaluation.png"
)

print("\n Benchmark Finished Successfully ")

