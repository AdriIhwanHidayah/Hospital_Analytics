from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import numpy as np

def train_classifier(df):
    # Daftarkan nama fitur untuk keperluan grafik Feature Importance nanti
    feature_names = [
        "Age",
        "Billing Amount",
        "Stay_Duration",
        "Insurance_Encoded",
        "Condition_Encoded"
    ]

    # PERBAIKAN: Menggunakan .to_numpy(dtype=np.float32) untuk X agar PyArrow terkonversi sempurna ke NumPy
    X = df[feature_names].to_numpy(dtype=np.float32)
    
    # PERBAIKAN: Mengonversi y menjadi array string/object NumPy murni untuk menghindari error indexing PyArrow
    y = df["Medication"].astype(str).to_numpy()

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
    report = classification_report(y_test, pred, output_dict=True)

    # Kita kembalikan feature_names agar file frontend bisa menggambar grafik dengan benar
    return model, feature_names, y_test, pred, accuracy, cm, report