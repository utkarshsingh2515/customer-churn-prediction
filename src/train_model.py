from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
from data_preprocessing import preprocess_data


def train_model():
    X_train, X_test, y_train, y_test, scaler = preprocess_data()

    model = Sequential()
    model.add(Dense(11, activation="relu", input_dim=11))
    model.add(Dense(11, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=50, batch_size=32)

    # Save model and scaler
    model.save("models/churn_model.h5")
    joblib.dump(scaler, "models/scaler.pkl")

    print("Model training complete and saved.")


if __name__ == "__main__":
    train_model()
