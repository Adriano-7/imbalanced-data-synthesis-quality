#!/usr/bin/env python
# coding: utf-8

# # EDA: Breast Cancer Winsconsin
# 
# This classic dataset, sourced from the UCI Machine Learning Repository, provides a digitized set of features from breast mass biopsies.
# 
# The features are computed from a digitized image of a **fine needle aspirate (FNA)** of a breast mass. They describe characteristics of the cell nuclei present in the image, such as their size, shape, and texture. For each image, ten real-valued features are computed, and for each of these features, the mean value, standard error, and "worst" (mean of the three largest values) are recorded. This results in a total of 30 features.

# In[30]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("Libraries imported successfully!")


# In[31]:


import kagglehub
from kagglehub import KaggleDatasetAdapter

file_path = "data.csv"

df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "uciml/breast-cancer-wisconsin-data",
    file_path,
)

print("Dataset loaded successfully from Kaggle!")
df.head()


# In[32]:


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

# In[33]:


df = df.drop(columns=['id', 'Unnamed: 32'])

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

print("Columns after dropping 'id' and 'Unnamed: 32':\n", df.columns)
print("\nDiagnosis value counts after encoding:\n", df['diagnosis'].value_counts())
df.head()


# In[34]:


df.describe().T


# In[35]:


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

# In[36]:


mean_features = df.columns[1:11] 

df[mean_features].hist(bins=20, figsize=(15, 8), layout=(3, 4), color='skyblue', edgecolor='black')
plt.suptitle('Distribution of Mean-Value Features', size=15)
plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()


# In[37]:


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

# In[38]:


# First 11 columns (diagnosis + 10 mean features)
corr_matrix = df.iloc[:, 0:11].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of Mean Features and Diagnosis', fontsize=16)
plt.show()


# - **`concave points_mean` (0.82)**, **`radius_mean` (0.73)**, and **`perimeter_mean` (0.74)** show a strong positive correlation with a malignant diagnosis
# 
# - (**`radius_mean`**, **`perimeter_mean`**, **`area_mean`**) are almost perfectly correlated. Maybe one of them would be sufficient for modeling. Due to this multicolinearity feature techniques like feature selection or dimensionality reduction (PCA) can be interestant.

# In[39]:


pairplot_features = ['diagnosis', 'radius_mean', 'texture_mean', 'concavity_mean', 'smoothness_mean', 'symmetry_mean']

sns.pairplot(df[pairplot_features], hue='diagnosis', palette='viridis', diag_kind='kde')
plt.suptitle('Pair Plot of Key Features', y=1.02, fontsize=18)
plt.show()


#  The distributions for benign (blue) and malignant (green) tumorsin the variables **`radius_mean`** and **`concavity_mean`** are clearly separated, whereas features like `texture_mean` show significant overlap.

# In[40]:


features_to_plot = df.columns[1:11]
df_plot = df.copy()

scaler = StandardScaler()
df_plot[features_to_plot] = scaler.fit_transform(df_plot[features_to_plot])

df_mean_benign = df_plot[df_plot['diagnosis'] == 0][features_to_plot].mean()
df_mean_malignant = df_plot[df_plot['diagnosis'] == 1][features_to_plot].mean()

fig = go.Figure()

# Benign Trace (Blue)
fig.add_trace(go.Scatterpolar(
    r=df_mean_benign.values,
    theta=features_to_plot,
    fill='toself',
    name='Benign (0)',
    line=dict(color='blue')
))

# Malignant Trace (Red)
fig.add_trace(go.Scatterpolar(
    r=df_mean_malignant.values,
    theta=features_to_plot,
    fill='toself',
    name='Malignant (1)',
    line=dict(color='red')
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[-1, 2] 
        )),
    showlegend=True,
    title={'text': "Radar Chart of Mean Feature Values by Diagnosis", 'x':0.5, 'font': {'size': 20}},
    template="plotly_white"
)

fig.show()


# In[41]:


features = df.columns.drop('diagnosis')
X = df[features]
y = df['diagnosis']

X_scaled = StandardScaler().fit_transform(X)

# Reduce the 30 features down to 3 principal components
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(
    data=principal_components, 
    columns=['PC1', 'PC2', 'PC3']
)

df_pca['diagnosis'] = y

explained_variance = pca.explained_variance_ratio_
print(f"Variance captured by PC1: {explained_variance[0]:.2%}")
print(f"Variance captured by PC2: {explained_variance[1]:.2%}")
print(f"Variance captured by PC3: {explained_variance[2]:.2%}")
print(f"Total variance captured by the first 3 components: {sum(explained_variance):.2%}")

fig = px.scatter_3d(
    df_pca,
    x='PC1',
    y='PC2',
    z='PC3',
    color='diagnosis',
    color_discrete_map={0: 'dodgerblue', 1: 'red'},
    symbol='diagnosis',
    opacity=0.8,
    hover_name=df_pca.index,
    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'}
)

fig.update_layout(
    title={
        'text': "3D PCA of Breast Cancer Dataset",
        'y':0.95,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': {'size': 20}
    },
    legend_title_text='Diagnosis',

)

fig.for_each_trace(lambda t: t.update(name = {'0':'Benign', '1':'Malignant'}[t.name]))

fig.show()

