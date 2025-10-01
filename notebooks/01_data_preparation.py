#!/usr/bin/env python
# coding: utf-8

# # 1. Data Loading and Initial Setup
# 
# This notebook prepares the Mammographic Mass dataset for machine learning experiments focused on class imbalance.
# 
# The process is as follows:
# 1.  Load and clean the raw data using MICE imputation.
# 2.  Create a single, hold-out test set.
# 3.  From the main training set, generate multiple training datasets with varying **imbalance ratios (IR)** by undersampling the majority class.
# 4.  For each IR, create a size-matched **control dataset** with the original class ratio.
# 5.  Preprocess and save all generated datasets.

# In[33]:


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.utils import resample
from pathlib import Path

# --- Configuration ---
RAW_PATH = Path("../data/raw/mammographic_mass.csv")
PROCESSED_PATH = Path("../data/processed/")
TARGET_FEATURE = 'Severity'
RANDOM_STATE = 42
# Define Majority:Minority ratios to generate for the experiment
IMBALANCE_RATIOS = [1, 5, 10, 20, 50, 100] 

# --- Setup ---
RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

print("Fetching and loading the raw dataset...")
mammographic_mass = fetch_ucirepo(id=161)
X_features = mammographic_mass.data.features
y_target = mammographic_mass.data.targets
df_raw = pd.concat([X_features, y_target], axis=1)
df_raw.to_csv(RAW_PATH, index=False)
print(f"Raw dataset loaded and saved. Shape: {df_raw.shape}")


# # 2. Data Cleaning: Imputing Missing Values
# 
# As determined in the EDA, we use MICE to impute missing values to avoid introducing bias.

# In[34]:


cols_to_impute = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density']
imputer = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)

print("Starting MICE imputation...")
df_cleaned = df_raw.copy()
df_cleaned[cols_to_impute] = imputer.fit_transform(df_raw[cols_to_impute])

# Clip, round, and cast imputed values to their valid integer ranges
df_cleaned['BI-RADS'] = np.clip(df_cleaned['BI-RADS'], 1, 5)
df_cleaned['Shape'] = np.clip(df_cleaned['Shape'], 1, 4)
df_cleaned['Margin'] = np.clip(df_cleaned['Margin'], 1, 5)
df_cleaned['Density'] = np.clip(df_cleaned['Density'], 1, 4)
for col in cols_to_impute:
    df_cleaned[col] = df_cleaned[col].round().astype(int)

print(f"Data cleaned and imputed. Shape: {df_cleaned.shape}")
print("\nMissing values after imputation:", df_cleaned.isnull().sum().sum())


# # 3. Confirming Majority and Minority Classes
# #
# #For our imbalance experiments, we need to clearly define the majority and minority classes.
# #- **Class 0 (Majority):** Benign
# #- **Class 1 (Minority):** Malignant
# 

# In[35]:


print("Target variable distribution:")
print(df_cleaned[TARGET_FEATURE].value_counts())


# # 4. Create a Hold-Out Test Set
# 
# We perform a one-time stratified split to create a final test set. All experimental datasets will be generated from the `train_full_df`.
# 

# In[36]:


X = df_cleaned.drop(TARGET_FEATURE, axis=1)
y = df_cleaned[[TARGET_FEATURE]]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

train_full_df = pd.concat([X_train_full, y_train_full], axis=1)

print(f"Full training set shape: {train_full_df.shape}")
print(f"Hold-out test set shape: {X_test.shape}")


# # 5. Generate Imbalanced and Control Datasets (Corrected Logic)
# 
# 1.  Start with the **full majority class** ('Benign').
# 2.  **Undersample the minority class** ('Malignant') to achieve the desired Imbalance Ratio (IR).
# 3.  Create a size-matched **control dataset** for each IR.

# 
#   * **Methodology:** "To generate datasets with varying degrees of class imbalance, the majority class was held constant at 412 samples while the minority class was progressively undersampled to achieve imbalance ratios from 5:1 to 100:1. It should be noted that this approach intrinsically links a higher imbalance ratio with a smaller number of minority class samples."
#   * **Discussion:** When interpreting your results, we can't claim that the degradation in synthetic data quality is *only* due to the imbalance ratio. 

# In[37]:


malignant_df = train_full_df[train_full_df[TARGET_FEATURE] == 1]
benign_df = train_full_df[train_full_df[TARGET_FEATURE] == 0]
n_minority_available = len(malignant_df)
n_majority_available = len(benign_df)

print(f"Full training set composition: {n_majority_available} majority (Benign), {n_minority_available} minority (Malignant).")

generated_datasets = {}

for ir in IMBALANCE_RATIOS:
    print(f"\nProcessing Imbalance Ratio (IR) = {ir}:1")
    majority_full_set = benign_df
    
    # Calculate how many minority samples are needed to achieve the IR.
    n_minority_imbalanced = int(n_majority_available / ir)

    # Sanity check: ensure we have enough minority samples and don't sample zero.
    if n_minority_imbalanced > n_minority_available:
        print(f"  - SKIPPING: Cannot create {ir}:1 ratio as it requires more minority samples than available.")
        continue
    if n_minority_imbalanced < 1:
        print(f"  - SKIPPING: Ratio {ir}:1 results in zero minority samples.")
        continue
        
    # Undersample the minority class to the calculated size.
    minority_undersampled = resample(
        malignant_df,
        replace=False,
        n_samples=n_minority_imbalanced,
        random_state=RANDOM_STATE
    )
    
    # Create the new imbalanced dataset
    imbalanced_df = pd.concat([majority_full_set, minority_undersampled])
    total_size = len(imbalanced_df)
    generated_datasets[f'imbalanced_ir_{ir}'] = imbalanced_df
    print(f"  - Imbalanced set created: {total_size} samples ({len(majority_full_set)} majority, {len(minority_undersampled)} minority)")

    # Create the size-matched control dataset
    if total_size >= len(train_full_df):
        control_df = train_full_df.copy()
    else:
        control_df, _ = train_test_split(
            train_full_df,
            train_size=total_size,
            random_state=RANDOM_STATE,
            stratify=train_full_df[TARGET_FEATURE]
        )
    
    generated_datasets[f'control_ir_{ir}'] = control_df
    print(f"  - Control set created:      {len(control_df)} samples (original class ratio)")


# # 6. Preprocessing and Saving All Datasets
# 
# We fit the scaler **once** on the full training data. Then, we transform all generated training sets and the hold-out test set using this single, consistent scaler.
# 

# In[38]:


FEATURES_TO_SCALE = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density']

# Fit scaler on the complete training data to learn population parameters
scaler = StandardScaler()
scaler.fit(X_train_full[FEATURES_TO_SCALE])

# Process and save each generated training dataset
for name, df in generated_datasets.items():
    X_temp = df.drop(columns=[TARGET_FEATURE])
    y_temp = df[[TARGET_FEATURE]]

    # Apply the pre-fitted scaler
    X_processed = scaler.transform(X_temp[FEATURES_TO_SCALE])
    X_processed_df = pd.DataFrame(X_processed, columns=FEATURES_TO_SCALE)
    
    # Combine processed features and target
    final_df = X_processed_df.reset_index(drop=True)
    final_df[TARGET_FEATURE] = y_temp.reset_index(drop=True)
    
    save_path = PROCESSED_PATH / f"train_{name}.csv"
    final_df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

# Finally, process and save the hold-out test set
X_test_processed = scaler.transform(X_test[FEATURES_TO_SCALE])
X_test_processed_df = pd.DataFrame(X_test_processed, columns=FEATURES_TO_SCALE)

test_df = X_test_processed_df.reset_index(drop=True)
test_df[TARGET_FEATURE] = y_test.reset_index(drop=True)
test_df.to_csv(PROCESSED_PATH / "test.csv", index=False)
print(f"Saved: {PROCESSED_PATH / 'test.csv'}")

print("\nPreprocessing complete. All datasets are ready for experiments.")

