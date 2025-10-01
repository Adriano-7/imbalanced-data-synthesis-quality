#!/usr/bin/env python
# coding: utf-8

# # 1. Data Loading and Initial Setup
# 
# This notebook prepares the Maternal Health Risk dataset for our experiments. We will perform the following steps:
# 1.  Load the raw data.
# 2.  Apply the cleaning steps identified in the EDA (removing biased duplicates and outliers).
# 3.  Split the data into training and a hold-out test set to prevent data leakage.
# 4.  Apply preprocessing (scaling and encoding) fitted *only* on the training data.
# 5.  Save the final, analysis-ready datasets to `/data/processed/`.

# In[4]:


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from pathlib import Path


# In[5]:


RAW_PATH = Path("../data/raw/maternal_health_risk.csv")
PROCESSED_PATH = Path("../data/processed/")
TARGET_FEATURE = 'RiskLevel'
NUMERICAL_FEATURES = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']

RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

maternal_health_risk = fetch_ucirepo(id=863)
X_features = maternal_health_risk.data.features
y_target = maternal_health_risk.data.targets
df_raw = pd.concat([X_features, y_target], axis=1)
df_raw.to_csv(RAW_PATH, index=False)

print("Raw dataset:")
display(df_raw.head())


# # 2. Data Cleaning
# 
# Based on the findings from our EDA, we must perform two critical cleaning steps.
# 
# ### 2.1. Removing Biased Duplicates
# The raw dataset contains 562 duplicate rows. Our EDA revealed that these duplicates were not random, but were heavily skewed towards 'mid risk' and 'high risk' profiles. Keeping them would introduce significant bias into our models. Therefore, we remove them.
# 

# In[6]:


df_cleaned = df_raw.drop_duplicates().copy()
print(f"Shape after removing duplicates: {df_cleaned.shape}")


# ### 2.2. Removing Physiological Outlier
# The EDA also identified a record with a `HeartRate` of 7 bpm, which is physiologically impossible for a living person and is clearly a data entry error. We remove this record to maintain data integrity.
# 

# In[7]:


df_cleaned = df_cleaned[df_cleaned['HeartRate'] != 7].reset_index(drop=True)
print(f"Final shape after cleaning: {df_cleaned.shape}")


# # 3. Data Splitting
# 
# Let's create a hold-out test set **before** applying any further transformations. This test set will not be touched during training and will serve as our final, unbiased benchmark for all models (baseline and those trained on synthetic data). We use stratified splitting to maintain the same class distribution in both the training and test sets.
# 

# In[8]:


X = df_cleaned.drop(TARGET_FEATURE, axis=1)
y = df_cleaned[[TARGET_FEATURE]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")


# # 4. Data Preprocessing
# 
# Machine learning models require numerical input and often perform better when features are on a similar scale. We will apply two transformations:
# 
# * **Ordinal Encoding**: The target variable `RiskLevel` has an inherent order ('low risk' < 'mid risk' < 'high risk'). We will convert these categories into numerical values (0, 1, 2) to reflect this.
# * **Standard Scaling**: We will scale all numerical features to have a mean of 0 and a standard deviation of 1. This is essential for distance-based and gradient-based algorithms.
# 
# **Crucially**, the scaler will be `fit` only on the `X_train` data and then used to `transform` both `X_train` and `X_test`. This prevents any information from the test set leaking into our training process.
# 

# In[9]:


# Target Encoding
target_encoder = OrdinalEncoder(categories=[['low risk', 'mid risk', 'high risk']])
y_train_processed = target_encoder.fit_transform(y_train)
y_test_processed = target_encoder.transform(y_test)

# Feature Scaling
scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
X_test_processed = scaler.transform(X_test[NUMERICAL_FEATURES])


# # 5. Saving the Final Datasets
# 
# Finally, we combine the processed features and targets into final DataFrames and save them as CSV files. The `experiment_runner` script will load these files directly, ensuring a clean separation of concerns.
# 

# In[10]:


train_df = pd.DataFrame(X_train_processed, columns=NUMERICAL_FEATURES)
train_df[TARGET_FEATURE] = y_train_processed

test_df = pd.DataFrame(X_test_processed, columns=NUMERICAL_FEATURES)
test_df[TARGET_FEATURE] = y_test_processed

train_df.to_csv(PROCESSED_PATH / "train.csv", index=False)
test_df.to_csv(PROCESSED_PATH / "test.csv", index=False)

print("Processed training and testing sets saved successfully!")
print("\nProcessed Training Data Head:")
display(train_df.head())

