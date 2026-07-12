from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def train_classifier(df):

    X = df[
        [
            "Age",
            "Billing Amount",
            "Stay_Duration",
            "Insurance_Encoded"
        ]
    ]

    y = df["Medical Condition"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, pred)

    cm = confusion_matrix(y_test, pred)

    report = classification_report(
        y_test,
        pred,
        output_dict=True
    )

    return model, X_test, y_test, pred, accuracy, cm, report