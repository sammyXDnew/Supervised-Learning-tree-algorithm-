import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset with proper delimiter
dataset = pd.read_csv(
    "D:\\UNTUK KULIAH SEMESTER 4\\Pembelajaran Mesin A\\CODE\\Pertemuan 7\\WEC_Perth_49_1.csv", 
    delimiter=';'
)

# Ensure columns are properly parsed
print(dataset.head())
print(dataset.info())

# Exploratory Data Analysis
dataset.head()
dataset.info()
dataset.describe()
g = sns.pairplot(dataset, hue='Total_Power', diag_kind='hist')  # Menambahkan diag_kind='hist' untuk menampilkan histogram
g.fig.suptitle("Scatterplot and histogram of pairs of variables color coded by Total_Power", 
               fontsize = 14, # defining the size of the title
               y=1.05); # y = defining title y position (height)

# Preprocessing (assuming 'Total_Power' is the target variable)
if 'Total_Power' in dataset.columns:
    X = dataset.drop('Total_Power', axis=1)  # Features
    y = dataset['Total_Power']  # Target
else:
    print("Error: 'Total_Power' column not found in the dataset.")

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))