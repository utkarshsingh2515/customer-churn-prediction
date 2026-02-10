import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("models/churn_model.h5")

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

scaler = joblib.load("models/scaler.pkl")

def predict_churn(features):
    features = scaler.transform(np.array([features]))
    probability = model.predict(features)[0][0]
    if probability > 0.5:
        print("Customer likely to churn")
    else:
        print("Customer likely to stay")
    return probability

print(predict_churn([1,0,500,30,2,1,1,100000,1,0,1]))
