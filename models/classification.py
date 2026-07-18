from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Tambahkan parameter _force_refresh agar cache otomatis jebol jika nilainya berubah
def train_classifier(df, _force_refresh=None):
    # Daftarkan nama fitur murni dalam list
    feature_names = [
        "Age",
        "Billing Amount",
        "Stay_Duration",
        "Insurance_Encoded",
        "Condition_Encoded"
    ]

    # Ambil data dalam bentuk DataFrame murni (bukan NumPy Array)
    X = df[feature_names].copy()
    
    # Target (Medication) dikonversi ke tipe string biasa demi keamanan backend data
    y = df["Medication"].astype(str)

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

    # Mengembalikan objek model dan kelengkapannya
    return model, feature_names, y_test, pred, accuracy, cm, report