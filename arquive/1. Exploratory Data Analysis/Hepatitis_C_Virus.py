#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis: Hepatitis C Virus (HCV) in Egyptian Patients
# 
# The "Hepatitis C Virus (HCV) for Egyptian patients" dataset from the UCI Machine Learning Repository contains clinical and demographic data for Egyptian patients who underwent treatment for HCV over 18 months. It includes a variety of features, from basic demographics and symptoms to complex blood work results like liver enzyme levels and RNA measurements.
# 
# The target variable for our analysis is the **Baseline Histological Staging**, which represents the severity of liver fibrosis, graded from F1 to F4 (Cirrhosis).

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)

print("Libraries imported successfully!")


# In[17]:


import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

file_path = "HCV-Egy-Data.csv"

df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "nourmibrahim/hepatitis-c-virus-hcv-for-egyptian-patients",
  file_path,
)

print("First 5 rows of the dataset:")
display(df.head())

print(f"\nDataset shape: {df.shape}\n")
print("Dataset Information:")
df.info()



# * **Dataset Size:** The dataset contains **1385 records (patients)** and **29 columns (features)**. This is a moderately sized dataset, sufficient for initial modeling.
# 
# * **Data Completeness:** All 29 columns have **1385 non-null values**. This means there are **no missing values** in the dataset, which significantly simplifies the data cleaning process.
# 
# * **Data Types:** All features are of a numerical data type (`int64` or `float64`). This indicates that categorical features like `Gender` and binary symptoms (`Fever`, `Jaundice`, etc.) have already been **numerically encoded** (e.g., as 1 and 2).

# In[18]:


# Clean whitespace from all column names
df.columns = df.columns.str.strip()

# Decode Gender (Assuming 1: Male, 2: Female)
df['Gender'] = df['Gender'].map({1: 'Male', 2: 'Female'})

# Decode all binary symptom columns (Assuming 1: Absent, 2: Present)
symptom_cols = [
    'Fever', 
    'Nausea/Vomting', 
    'Headache',
    'Diarrhea', 
    'Fatigue & generalized bone ache', 
    'Jaundice',
    'Epigastric pain'
]

# This loop will now execute perfectly
for col in symptom_cols:
    df[col] = df[col].map({1: 'Absent', 2: 'Present'})

# Map the target variable to its clinical meaning
df['Fibrosis_Stage_Label'] = df['Baselinehistological staging'].map({
    1: 'F1 (Portal Fibrosis)',
    2: 'F2 (Few Septa)',
    3: 'F3 (Many Septa)',
    4: 'F4 (Cirrhosis)'
})

print("Data successfully decoded for categorical features.")


# Statistics
print("\nDescriptive Statistics for Numerical Features:")
display(df.describe().drop(columns=['Baselinehistological staging']).T.style.background_gradient(cmap='viridis'))




# In[19]:


fig = px.pie(
    df, 
    names='Fibrosis_Stage_Label', 
    title='Distribution of Liver Fibrosis Stages (Target Variable)',
    hole=0.3,
    color_discrete_sequence=px.colors.sequential.Plasma_r
)

fig.update_traces(
    textposition='inside', 
    textinfo='percent+label',
    hovertemplate='Stage: %{label}<br>Count: %{value}<br>Percentage: %{percent}'
)
fig.update_layout(showlegend=False)
fig.show()


# In the chart we can see that the target variable is **well-balanced**, with each of the four fibrosis stages representing about 25% of the data.

# In[20]:


print(df.columns)


# In[21]:


fig = make_subplots(
    rows=1, cols=3,
    specs=[[{'type':'xy'}, {'type':'xy'}, {'type':'domain'}]],
    subplot_titles=('Age Distribution', 'BMI Distribution', 'Gender Distribution')
)

# Age Histogram (xy type)
fig.add_trace(go.Histogram(x=df['Age'], name='Age', marker_color='#636EFA'), row=1, col=1)

# BMI Histogram (xy type)
fig.add_trace(go.Histogram(x=df['BMI'], name='BMI', marker_color='#EF553B'), row=1, col=2)

# Gender Pie Chart (domain type)
gender_counts = df['Gender'].value_counts()
fig.add_trace(go.Pie(labels=gender_counts.index, values=gender_counts.values, name='Gender', hole=0.3), row=1, col=3)

fig.update_layout(title_text='Patient Demographics Analysis', showlegend=False)
fig.show()


# The patient cohort is well-distributed without significant skews.
# 
# * **Age & BMI:** Both show a relatively **uniform distribution** across their respective ranges (Age: 32-61, BMI: 22-35). This means the dataset isn't biased towards a specific age or weight group.
# * **Gender:** The gender split is **perfectly balanced** at nearly 50/50.
# 

# In[22]:


# Symptom Prevalence Analysis
symptom_df = df[symptom_cols].apply(lambda x: x.value_counts()).T.loc[:, 'Present'].sort_values(ascending=False)
fig2 = px.bar(
    symptom_df,
    x=symptom_df.index,
    y=symptom_df.values,
    title='Prevalence of Symptoms Among Patients',
    labels={'x': 'Symptom', 'y': 'Number of Patients Presenting Symptom'},
    color=symptom_df.values,
    color_continuous_scale='sunset'
)
fig2.show()


# ### Symptom Prevalence
# 
# Each symptom is present in roughly half the patients (~700 out of 1385).
# Because no single symptom stands out, they are likely **weak individual predictors**. They don't create clear subgroups within the data.

# In[23]:


# Bivariate Analysis: Continuous Features vs. Fibrosis Stage
continuous_features = ['Age', 'BMI', 'RNA Base', 'WBC', 'RBC', 'Plat']

# Create a 2x3 grid of subplots
fig = make_subplots(
    rows=2, cols=3,
    subplot_titles=continuous_features
)

# Loop through the features and add a Box plot for each
for i, feature in enumerate(continuous_features):
    row, col = (i // 3) + 1, (i % 3) + 1

    # Add a Plotly Box trace instead of a Seaborn plot
    fig.add_trace(
        go.Box(
            y=df[feature], 
            x=df['Fibrosis_Stage_Label'], # Use the readable label
            name=feature,
            marker_color=px.colors.qualitative.Vivid[i] # Assign a color
        ),
        row=row, col=col
    )

# Update the layout for a clean look
fig.update_layout(
    title_text='Continuous Features vs. Liver Fibrosis Stage',
    showlegend=False, # Hide legend as titles are enough
    height=700 # Adjust height for better spacing
)

fig.show()


# In[24]:


numerical_df = df.select_dtypes(include=np.number)

corr_matrix = numerical_df.corr()

plt.figure(figsize=(20, 16))
sns.heatmap(
    corr_matrix, 
    annot=True, 
    fmt='.2f', 
    cmap='coolwarm', 
    linewidths=0.5
)
plt.title('Correlation Matrix of Numerical Features', fontsize=20)
plt.show()


# In[25]:


alt_cols = ['ALT 1', 'ALT4', 'ALT 12', 'ALT 24', 'ALT 36', 'ALT 48']
alt_df = df.groupby('Fibrosis_Stage_Label')[alt_cols].mean().T
alt_df.index = [1, 4, 12, 24, 36, 48] # Set index to weeks for plotting

# RNA Levels Over Time
rna_cols = ['RNA Base', 'RNA 4', 'RNA 12', 'RNA EOT', 'RNA EF']
rna_df = df.groupby('Fibrosis_Stage_Label')[rna_cols].mean().T
rna_df.index = [0, 4, 12, 24, 52] # Rough week mapping: EOT~24w, EF~52w

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

# ALT Plot
alt_df.plot(kind='line', marker='o', ax=ax1)
ax1.set_title('Average ALT Levels During Treatment by Fibrosis Stage', fontsize=16)
ax1.set_xlabel('Weeks of Treatment')
ax1.set_ylabel('Mean ALT Level')
ax1.legend(title='Fibrosis Stage')
ax1.grid(True)

# RNA Plot
rna_df.plot(kind='line', marker='x', ax=ax2)
ax2.set_title('Average RNA Levels During Treatment by Fibrosis Stage', fontsize=16)
ax2.set_xlabel('Weeks of Treatment')
ax2.set_ylabel('Mean RNA (Viral Load)')
ax2.legend(title='Fibrosis Stage')
ax2.grid(True)

plt.tight_layout()
plt.show()


# ## Part 2: Creation of Imbalanced Datasets for Comparative Analysis
# 
# The initial EDA revealed that the publicly available HCV dataset is remarkably **well-balanced**, with each of the four fibrosis stages (F1-F4) representing approximately 25% of the cohort. While this is convenient for standard modeling, our primary research goal is to evaluate how synthetic data generation algorithms perform under the stress of **class imbalance**, a condition far more typical in real-world medical datasets where advanced disease stages are often rare.
# 
# To achieve this, we will systematically engineer a series of datasets derived from the original data. Each new dataset will feature a controlled, escalating **Imbalance Ratio (IR)**, defined as the ratio of the number of samples in the majority class to the number of samples in a minority class.
# 
# ### Methodology
# 
# Our approach is designed to isolate the effect of the Imbalance Ratio itself:
# 
# 1.  **Baseline (IR=1):** The original, balanced dataset of 1385 patients will serve as our baseline control group.
# 2.  **Majority Class Designation:** We will designate the most severe stage, **`F4 (Cirrhosis)`**, as the single majority class. This is a clinically relevant scenario, as we want to test the algorithms' ability to generate plausible data for the less severe (and now, artificially rarer) stages. The number of F4 samples will remain constant across all generated datasets.
# 3.  **Minority Class Undersampling:** The other three stages, **`F1`**, **`F2`**, and **`F3`**, will be treated as minority classes. To achieve a target Imbalance Ratio (e.g., IR=10), we will randomly undersample these three classes so that each has approximately 1/10th the number of samples as the F4 class.
# 4.  **Controlled Escalation:** We will generate datasets with the following target Imbalance Ratios: **IR = 2, 5, 10, and 20**.
# 5.  **Reproducibility:** A fixed `random_state` will be used throughout the sampling process to ensure that our results are fully reproducible.
# 
# This systematic process will yield a collection of datasets where the only significant variable changing is the class distribution, allowing for a rigorous and unbiased comparison of synthetic data generation techniques in the subsequent stages of our research.
# becomes more severe.

# In[26]:


from sklearn.utils import resample
import warnings
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_imbalanced_dataset(df_source, target_column, minority_class_label, n_minority_samples, random_state=42):
    df_minority = df_source[df_source[target_column] == minority_class_label]
    df_majority = df_source[df_source[target_column] != minority_class_label]

    df_minority_downsampled = resample(
        df_minority,
        replace=False,  # Sample without replacement
        n_samples=n_minority_samples,
        random_state=random_state
    )

    df_imbalanced = pd.concat([df_majority, df_minority_downsampled])

    df_imbalanced = df_imbalanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df_imbalanced


# In[27]:


TARGET_VARIABLE = 'Fibrosis_Stage_Label'
MINORITY_CLASS = 'F4 (Cirrhosis)'
TARGET_IRS = [1, 2, 5, 10, 20, 50] 

imbalanced_datasets = {}

imbalanced_datasets[1] = df.copy() 


# In[28]:


majority_counts = df[df[TARGET_VARIABLE] != MINORITY_CLASS][TARGET_VARIABLE].value_counts()
n_majority_reference = majority_counts.max()
print(f"Reference Majority Class Size (for F3): {n_majority_reference} samples.\n")


# In[29]:


# 3. Generate datasets for each target IR > 1
for ir in TARGET_IRS[1:]:
    n_minority = max(1, int(n_majority_reference / ir))
    print(f"Generating dataset for Target IR={ir} with {n_minority} samples for '{MINORITY_CLASS}'...")

    df_new = create_imbalanced_dataset(df, TARGET_VARIABLE, MINORITY_CLASS, n_minority)
    imbalanced_datasets[ir] = df_new

print("\n Dataset Generation Complete ")


# In[30]:


n_datasets = len(TARGET_IRS)
fig = make_subplots(
    rows=2, cols=3,
    specs=[[{'type':'domain'}]*3, [{'type':'domain'}]*3],
    subplot_titles=[f"<b>IR ≈ {ir}</b>" for ir in TARGET_IRS]
)

for i, ir in enumerate(TARGET_IRS):
    row, col = (i // 3) + 1, (i % 3) + 1

    temp_df = imbalanced_datasets[ir]
    class_counts = temp_df[TARGET_VARIABLE].value_counts()

    # Calculate the actual IR
    actual_ir = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')

    fig.add_trace(go.Pie(
        labels=class_counts.index, 
        values=class_counts.values, 
        name=f'IR={ir}'
    ), row=row, col=col)

    fig.layout.annotations[i]['text'] = (
        f"<b>Target IR: {ir}</b><br>"
        f"Actual IR: {actual_ir:.1f}<br>"
        f"Total Samples: {len(temp_df)}"
    )

fig.update_traces(
    hole=.4, 
    hoverinfo="label+percent+value",
    textinfo='percent',
    textfont_size=12
)
fig.update_layout(
    title_text="<b>Class Distribution Across Systematically Generated Imbalanced Datasets</b>",
    height=700,
    showlegend=False,
    font=dict(family="Arial, sans-serif", size=12)
)
fig.show()


# In[32]:


print("\nHead of the most imbalanced dataset (IR=50):")
display(imbalanced_datasets[50].head())
print("\nClass counts for the most imbalanced dataset (IR=50):")
print(imbalanced_datasets[50][TARGET_VARIABLE].value_counts())


# In[34]:


import os
output_dir = '../Imbalanced_Datasets/Hepatitis_C_Virus/'

os.makedirs(output_dir, exist_ok=True)

print(f"Directory '{output_dir}' is ready.")

for ir, df_to_save in imbalanced_datasets.items():
    filename = f"hcv_dataset_ir_{ir}.csv"
    full_path = os.path.join(output_dir, filename)

    df_to_save.to_csv(full_path, index=False)

    print(f"Successfully saved: {full_path}")

print("\n All datasets have been saved. ")

