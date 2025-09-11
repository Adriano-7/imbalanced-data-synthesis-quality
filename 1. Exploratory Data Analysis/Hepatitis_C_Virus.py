#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis: Hepatitis C Virus (HCV) in Egyptian Patients
# 
# The "Hepatitis C Virus (HCV) for Egyptian patients" dataset from the UCI Machine Learning Repository contains clinical and demographic data for Egyptian patients who underwent treatment for HCV over 18 months. It includes a variety of features, from basic demographics and symptoms to complex blood work results like liver enzyme levels and RNA measurements.
# 
# The target variable for our analysis is the **Baseline Histological Staging**, which represents the severity of liver fibrosis, graded from F1 to F4 (Cirrhosis).

# In[10]:


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


# In[11]:


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

# In[12]:


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




# In[13]:


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

# In[14]:


print(df.columns)


# In[15]:


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

# In[16]:


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

# In[17]:


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


# In[18]:


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


# In[19]:


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






