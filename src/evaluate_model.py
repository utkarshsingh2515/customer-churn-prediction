import tensorflow 
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_preprocessing import preprocess_data
import numpy as np

# Load data
X_train, X_test, y_train, y_test, _ = preprocess_data()

# Load model
model = load_model("models/churn_model.h5")

# Predictions
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

