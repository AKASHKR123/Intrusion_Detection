import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for preprocessing, modeling, and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve
)

# SMOTE for oversampling to handle imbalanced datasets
from imblearn.over_sampling import SMOTE

# Classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# ------------------------------------------------------------------------------
# Load training and testing datasets
# ------------------------------------------------------------------------------
training_data_path = r"C:\Users\Akash\Documents\Intrusion_Detection\Dataset\UNSW_NB15_training-set.csv"
testing_data_path = r"C:\Users\Akash\Documents\Intrusion_Detection\Dataset\UNSW_NB15_testing-set.csv"

df_train = pd.read_csv(training_data_path)
df_test = pd.read_csv(testing_data_path)

# ------------------------------------------------------------------------------
# Combine both datasets for unified analysis
# ------------------------------------------------------------------------------
df = pd.concat([df_train, df_test], axis=0)

# ------------------------------------------------------------------------------
# Basic dataset inspection
# ------------------------------------------------------------------------------
print(f"Number of rows in the combined dataset: {df.shape[0]}")
print(f"Number of columns in the combined dataset: {df.shape[1]}")
print("\nData types and non-null counts:")
print(df.info())

# ------------------------------------------------------------------------------
# View the first few rows of the dataset
# ------------------------------------------------------------------------------
pd.set_option('display.max_columns', None)  # Show all columns when printing
print("\nFirst 5 rows of the dataset:")
print(df.head())

# ------------------------------------------------------------------------------
# Check for missing values in each column
# ------------------------------------------------------------------------------
missing_values = df.isnull().sum().to_dict()
print("\nMissing values in each column:")
for col, val in missing_values.items():
    print(f"{col}: {val}")

# ------------------------------------------------------------------------------
# Check for duplicate rows
# ------------------------------------------------------------------------------
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# ------------------------------------------------------------------------------
# Display unique value counts for each column
# ------------------------------------------------------------------------------
print("\nUnique value counts per column:")
for column in df.columns:
    unique_vals = df[column].value_counts()
    print(f"\nUnique values in '{column}' ({len(unique_vals)} unique values):")
    print(unique_vals)
