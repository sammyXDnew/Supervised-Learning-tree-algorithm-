from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load dataset
data = pd.read_csv('WEC_Perth_49_1.csv')  # Replace with your dataset file
X = data.drop('Total_Power', axis=1)  # Replace 'target' with your target column name
y = data['Total_Power']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42, use_label_encoder=False)
model.fit(X_train, y_train, eval_metric='logloss')

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {accuracy}")

# Additional evaluation metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
