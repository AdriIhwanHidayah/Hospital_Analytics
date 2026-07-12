from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import numpy as np


def train_regression(df):

    X = df[
        [
            "Age",
            "Stay_Duration",
            "Condition_Encoded",
            "Insurance_Encoded"
        ]
    ]

    y = df["Estimated_Billing"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)

    mse = mean_squared_error(y_test, pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, pred)

    return (
        model,
        X_test,
        y_test,
        pred,
        mae,
        mse,
        rmse,
        r2
    )