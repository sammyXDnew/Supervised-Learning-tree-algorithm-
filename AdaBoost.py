import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

# Load dataset from CSV file
file_path = r'd:\\UNTUK KULIAH SEMESTER 4\\Pembelajaran Mesin A\\CODE\\Pertemuan 7\\WEC_Perth_49_1.csv'
data = pd.read_csv(file_path)

# Ensure the dataset is not empty and has the correct structure
if data.empty:
    raise ValueError("The dataset is empty. Please check the file path or the dataset content.")
if data.shape[1] < 2:
    raise ValueError("The dataset must have at least one feature column and one target column.")

# Assuming the last column is the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Check if X and y are valid
if X.shape[1] == 0:
    raise ValueError("No feature columns found in the dataset. Please check the dataset structure.")
if len(y) == 0:
    raise ValueError("No target values found in the dataset. Please check the dataset structure.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = AdaBoostClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Evaluate the model on the testing data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % accuracy)

# Make a single prediction (example row, replace with actual data if needed)
row = [[10]]  # Replace with appropriate feature values
yhat = model.predict(row)
print('Predicted Class: %d' % yhat[0])