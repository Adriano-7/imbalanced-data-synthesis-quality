#!/usr/bin/env python
# coding: utf-8

# # Analyzing Generative Model Robustness to Class Imbalance

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from sklearn.utils import shuffle

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries imported successfully!")


# In[5]:


file_path = "data.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
df.head()


# In[6]:


print(f"Dataset shape: {df.shape}\n")

print("Dataset Info:")
df.info()


# By observing the dataset info we get the following info:
# - The dataset contains 569 entries and 33 columns.
# - The id column is just an identifier and Unnamed: 32 is completely empty (NaN) so we'll drop it.
# - The diagnosis column is of type object and will need to be converted to a numerical format.
# - There are no missing values in the feature columns, which simplifies our preprocessing.
# 
# 

# In[7]:


# Drop the unnecessary 'id' and 'Unnamed: 32' columns
df = df.drop(columns=['id', 'Unnamed: 32'])

# Encode the 'diagnosis' column: M (malignant) -> 1, B (benign) -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("Columns after dropping 'id' and 'Unnamed: 32':\n", df.columns)
print("\nDiagnosis value counts after encoding:\n", df['diagnosis'].value_counts())
df.head()


# In[8]:


df.describe().T


# In[9]:


plt.figure(figsize=(7, 5))
ax = sns.countplot(x='diagnosis', data=df, palette='viridis')

plt.title('Distribution of Diagnosis (1: Malignant, 0: Benign)', fontsize=16)
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Benign (0)', 'Malignant (1)'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.show()


# - The dataset contains 357 benign cases and 212 malignant cases.
# 
# - The classes are somewhat imbalanced but not severely so. 

# In[10]:


mean_features = df.columns[1:11] 

df[mean_features].hist(bins=20, figsize=(15, 8), layout=(3, 4), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Mean-Value Features', size=15)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()


# In[11]:


df_melted = pd.melt(df, id_vars="diagnosis", value_vars=mean_features, var_name="feature", value_name="value")

plt.figure(figsize=(10, 8))
sns.boxplot(x="feature", y="value", hue="diagnosis", data=df_melted, palette="Set2")
plt.xticks(rotation=45)
plt.title('Comparison of Mean Features for Benign (0) vs. Malignant (1) Tumors', fontsize=16)
plt.show()


# - Features like radius_mean, perimeter_mean, and area_mean are clearly valuable for building a predictive machine learning model.
# 
# - The difference in scales between features highlights why scaling should be perform. Without it, models that are sensitive to feature magnitude (like SVMs or Logistic Regression) will be biased by features with large values, like area_mean.
# 
# 

# In[12]:


# First 11 columns (diagnosis + 10 mean features)
corr_matrix = df.iloc[:, 0:11].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Mean Features and Diagnosis', fontsize=16)
plt.show()


# - **`concave points_mean` (0.82)**, **`radius_mean` (0.73)**, and **`perimeter_mean` (0.74)** show a strong positive correlation with a malignant diagnosis
# 
# - (**`radius_mean`**, **`perimeter_mean`**, **`area_mean`**) are almost perfectly correlated. Maybe one of them would be sufficient for modeling. Due to this multicolinearity feature techniques like feature selection or dimensionality reduction (PCA) can be interestant.

# In[13]:


pairplot_features = ['diagnosis', 'radius_mean', 'texture_mean', 'concavity_mean', 'smoothness_mean', 'symmetry_mean']

sns.pairplot(df[pairplot_features], hue='diagnosis', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Key Features', y=1.02, fontsize=18)
plt.show()


#  The distributions for benign (blue) and malignant (green) tumorsin the variables **`radius_mean`** and **`concavity_mean`** are clearly separated, whereas features like `texture_mean` show significant overlap.

# 
# ### Generating Controlled Imbalanced Datasets
# 
# To systematically evaluate how class imbalance affects the quality of synthetic data, we will generate a series of datasets with precisely controlled, escalating **Imbalance Ratios (IRs)**.
# 
# * **Constant Minority Class**: The number of minority class samples (**Malignant**, n=212) will be held constant across all experimental conditions. This ensures that the information available for the generator to learn from the minority class remains identical in every experiment.
# * **Majority Class Undersampling**: The majority class (**Benign**, n=357) will be randomly undersampled to create training sets that match our target IRs.
# 

# In[14]:


df_minority = df[df['diagnosis'] == 1].copy()
df_majority = df[df['diagnosis'] == 0].copy()

n_minority = len(df_minority)
n_majority_original = len(df_majority)
original_ir = n_majority_original / n_minority


# In[15]:


print(f"Original dataset composition:")
print(f"  - Minority (Malignant) samples: {n_minority}")
print(f"  - Majority (Benign) samples:    {n_majority_original}")
print(f"  - Maximum possible IR via undersampling: {original_ir:.2f}")
print("-" * 50)


# In[16]:


IMBALANCE_RATIOS = [1.0, 1.25, 1.5] 
RANDOM_STATE = 42
imbalanced_datasets = {}


# In[17]:


for ir in IMBALANCE_RATIOS:
    n_majority_target = int(n_minority * ir)

    if n_majority_target > n_majority_original:
        print(f"Skipping invalid IR={ir}")
        continue

    df_majority_sampled = df_majority.sample(n=n_majority_target, random_state=RANDOM_STATE)
    df_imbalanced = pd.concat([df_majority_sampled, df_minority])
    df_imbalanced = shuffle(df_imbalanced, random_state=RANDOM_STATE).reset_index(drop=True)
    imbalanced_datasets[ir] = df_imbalanced

    print(f"Successfully created dataset for IR = {ir:.2f}")
    print(f"  - Majority samples:   {len(df_majority_sampled)}")
    print(f"  - Minority samples:   {len(df_minority)}")
    print(f"  - Total samples:      {len(df_imbalanced)}")
    print("---")


# In[18]:


imbalanced_datasets[original_ir] = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)
print("Added the original dataset to the collection (IR ≈ 1.68)")
print("---")


# In[19]:


print(f"\nThe `imbalanced_datasets` dictionary now contains {len(imbalanced_datasets)} dataframes.")
print(f"Keys (IRs): {[round(k, 2) for k in imbalanced_datasets.keys()]}")


# In[20]:


import os

output_dir = "imbalanced_csvs"
os.makedirs(output_dir, exist_ok=True)

for ir, df_data in imbalanced_datasets.items():
    filename_ir = str(ir).replace('.', '_')
    file_path = os.path.join(output_dir, f"dataset_ir_{filename_ir}.csv")

    df_data.to_csv(file_path, index=False)

    print(f"Successfully saved dataset with IR={ir:.2f} to '{file_path}'")

print(f"\nAll imbalanced datasets have been saved to the '{output_dir}' directory.")


# # Train and evaluate

# In[ ]:


from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader

from sklearn.model_selection import train_test_split

from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import glob 
import re  


# In[ ]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Create a 70/30 split. 
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

df_train_full = pd.concat([X_train_full, y_train_full], axis=1)
df_test = pd.concat([X_test, y_test], axis=1)

print(f"Hold-out test set created. Shape: {df_test.shape}")
print("Test set class distribution:\n", df_test['diagnosis'].value_counts())


# In[36]:


df_minority_train = df_train_full[df_train_full['diagnosis'] == 1].copy()
df_majority_train = df_train_full[df_train_full['diagnosis'] == 0].copy()

n_minority_train = len(df_minority_train)

imbalanced_training_sets = {}
for ir in IMBALANCE_RATIOS:
    n_majority_target = int(n_minority_train * ir)

    df_majority_sampled = df_majority_train.sample(n=n_majority_target, random_state=RANDOM_STATE)
    df_imbalanced = pd.concat([df_majority_sampled, df_minority_train])
    df_imbalanced = shuffle(df_imbalanced, random_state=RANDOM_STATE).reset_index(drop=True)

    imbalanced_training_sets[ir] = df_imbalanced
    print(f"Created training set for IR = {ir:.2f} with {len(df_imbalanced)} samples.")


# Now, we'll iterate through each of our imbalanced training sets. For each one, we will train three different types of generative models (ctgan, tvae, adsgan) and evaluate them. The Benchmarks.evaluate function will:
# 
# - Train the generator on the provided imbalanced training data.
# - Use the trained generator to create synthetic data.
# - Train a downstream machine learning model (we'll use linear_model) on this synthetic data.
# - Evaluate the performance of the downstream model on our real hold-out test set.

# In[48]:


models_to_test = ["ctgan"]

test_loader = GenericDataLoader(df_test, target_column="diagnosis")
results = []

for ir, train_df in imbalanced_training_sets.items():
    print(f"\n# Benchmarking for Imbalance Ratio (IR) = {ir:.2f} #")

    train_loader = GenericDataLoader(train_df, target_column="diagnosis")
    tasks = [(f"{model}_{ir:.2f}", model, {}) for model in models_to_test]

    score = Benchmarks.evaluate(
        tests=tasks,         
        X=train_loader,       
        X_test=test_loader,   
        synthetic_size=len(train_df),
        metrics={"performance": ["linear_model"]},
        #repeats=3
        repeats=1
    )

    for model in models_to_test:
        test_name = f"{model}_{ir:.2f}"
        mean_aucroc = 0
        std_aucroc = 0

        if isinstance(score, dict):
            results_df_for_model = score[test_name]
            performance_row = results_df_for_model.loc['performance.linear_model.syn_ood']

            mean_aucroc = performance_row['mean']
            std_aucroc = performance_row['stddev']

        else: # It's a DataFrame 
            performance_row = score.loc[test_name]
            mean_aucroc = performance_row['performance.linear_model.syn_ood.mean']
            std_aucroc = performance_row['performance.linear_model.syn_ood.stddev']

        results.append({
            "ir": ir,
            "model": model,
            "mean_aucroc": mean_aucroc,
            "std_aucroc": std_aucroc,
            "all_aucrocs": [] 
        })

results_df = pd.DataFrame(results)

print("\n\n✅ Benchmark complete. Results are stored in results_df.")
results_df


# Let's now analyze how the performance (AUCROC) of each generative model changes as the class imbalance in the training data becomes more severe. 
# 
# A robust model will have a curve that remains relatively flat, while a fragile model will show a steep decline.
# 
# 

# In[49]:


plt.figure(figsize=(12, 7))

sns.lineplot(
    data=results_df,
    x='ir',
    y='mean_aucroc',
    hue='model',
    marker='o',
    style='model',
    markersize=8
)

plt.title('Generative Model Performance vs. Imbalance Ratio', fontsize=18, pad=20)
plt.xlabel('Imbalance Ratio (IR)', fontsize=14)
plt.ylabel('Mean AUCROC (on Real Test Data)', fontsize=14)
plt.legend(title='Generative Model', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.xticks(IMBALANCE_RATIOS)
plt.ylim(0.5, 1.0) 
plt.show()

