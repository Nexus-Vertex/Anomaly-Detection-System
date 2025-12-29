"""
Application Python simple pour détecter les anomalies dans un jeu de données,
avec génération de dataset, détection (IsolationForest + LOF) et visualisation.

Exécution :
    python app.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



def generate_data(n_samples=500, n_anomalies=20):
    rng = np.random.RandomState(42)
    normal_points = rng.normal(loc=0, scale=1, size=(n_samples, 2))
    anomalies = rng.uniform(low=-6, high=6, size=(n_anomalies, 2))
    data = np.vstack([normal_points, anomalies])
    df = pd.DataFrame(data, columns=["x", "y"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def preprocess(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df)
    return X


def detect_anomalies(X):
    results = {}

    # Isolation Forest
    isf = IsolationForest(contamination=0.05, random_state=0)
    pred_isf = isf.fit_predict(X)
    results["IsolationForest"] = (pred_isf == -1).astype(int)

    # Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    pred_lof = lof.fit_predict(X)
    results["LOF"] = (pred_lof == -1).astype(int)

    return pd.DataFrame(results)


def plot_results(df, preds):
    plt.figure(figsize=(7, 6))
    plt.scatter(df["x"], df["y"], s=25, label="Données normales")

    anom_isf = preds["IsolationForest"] == 1
    anom_lof = preds["LOF"] == 1

    plt.scatter(df["x"][anom_isf], df["y"][anom_isf], color="red", label="Anomalies (IF)")
    plt.scatter(df["x"][anom_lof], df["y"][anom_lof], marker="x", color="black", label="Anomalies (LOF)")

    plt.title("Détection d'anomalies")
    plt.legend()
    plt.tight_layout()
    plt.savefig("resultat.png")
    plt.close()



def main():
    print("Génération du jeu de données…")
    df = generate_data()

    print("Prétraitement…")
    X = preprocess(df)

    print("Détection des anomalies…")
    preds = detect_anomalies(X)

    final_df = pd.concat([df, preds], axis=1)

    print("Sauvegarde des résultats dans anomalies.csv…")
    final_df.to_csv("anomalies.csv", index=False)

    print("Génération du graphique resultat.png…")
    plot_results(df, preds)

    print("Terminé ! Fichiers générés : anomalies.csv et resultat.png")


if __name__ == "__main__":
    main()
