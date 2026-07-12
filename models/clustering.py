from sklearn.cluster import KMeans

def run_kmeans(df):

    X = df[["Age", "Billing Amount"]]

    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    df["Cluster"] = model.fit_predict(X)

    return df