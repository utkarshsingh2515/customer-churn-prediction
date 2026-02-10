import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data():
    # Load dataset
    df = pd.read_csv("data/Churn_Modelling.csv")

    # Drop unnecessary columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    # Split features and target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
