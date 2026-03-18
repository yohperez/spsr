"""
VitaBeats — Módulo 3: Red Neuronal (MLP + simulación RNN)
==========================================================
Detecta anomalías en series temporales de sensores del hogar
(uso de luz eléctrica y agua) que pueden indicar una caída,
una enfermedad aguda o inactividad prolongada.

Arquitectura:
    - Entrada: secuencia de 7 días × 2 sensores (luz + agua) = 14 features
    - Capa oculta 1: 64 neuronas, activación ReLU
    - Capa oculta 2: 32 neuronas, activación ReLU
    - Dropout: 0.3 para regularización
    - Salida: 1 neurona, activación Sigmoid → score de anomalía (0=normal, 1=anomalía)

Nota: Para un entorno académico sin GPU, se usa un MLP (sklearn) como
      aproximación al comportamiento de una RNN para series de 7 pasos.
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS (SERIES TEMPORALES)
# =============================================================================

def generate_sensor_dataset(n_samples: int = 800, random_state: int = 42) -> pd.DataFrame:
    """
    Genera series temporales sintéticas de 7 días para luz y agua.
    Patrón normal: variación gaussiana alrededor de la media personal.
    Patrón anómalo: caída brusca ≥50% en uno o más días (especialmente fin de semana).
    """
    np.random.seed(random_state)
    records = []

    for i in range(n_samples):
        # Media personal de cada sensor
        luz_mean   = np.random.uniform(7, 13)   # horas/día
        agua_mean  = np.random.uniform(35, 60)  # litros/día

        anomalia = i % 4 == 0  # 25% de anomalías para dataset balanceado

        luz_vals  = np.round(np.random.normal(luz_mean,  1.2, 7), 1)
        agua_vals = np.round(np.random.normal(agua_mean, 5.0, 7), 1)

        if anomalia:
            # Inyectar anomalía en uno o dos días (típicamente fin de semana)
            dias_anomalos = np.random.choice([5, 6], size=np.random.randint(1, 3), replace=False)
            for d in dias_anomalos:
                drop = np.random.uniform(0.55, 0.90)
                if np.random.rand() > 0.4:
                    agua_vals[d] = round(agua_vals[d] * (1 - drop), 1)
                if np.random.rand() > 0.4:
                    luz_vals[d]  = round(luz_vals[d]  * (1 - drop), 1)

        # Asegurar valores no negativos
        luz_vals  = np.maximum(luz_vals,  0)
        agua_vals = np.maximum(agua_vals, 0)

        record = {}
        for d in range(7):
            record[f"luz_dia{d+1}"]  = luz_vals[d]
            record[f"agua_dia{d+1}"] = agua_vals[d]
        record["anomalia"] = int(anomalia)
        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# 2. FEATURES ADICIONALES (ESTADÍSTICAS DE LA SERIE)
# =============================================================================

def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriquece el dataset con estadísticas de las series temporales:
    desviación estándar, caída máxima, ratio último_día/media.
    Estas features mejoran la capacidad de detección de la red.
    """
    luz_cols  = [f"luz_dia{i}"  for i in range(1, 8)]
    agua_cols = [f"agua_dia{i}" for i in range(1, 8)]

    df = df.copy()
    df["luz_std"]          = df[luz_cols].std(axis=1)
    df["agua_std"]         = df[agua_cols].std(axis=1)
    df["luz_mean"]         = df[luz_cols].mean(axis=1)
    df["agua_mean"]        = df[agua_cols].mean(axis=1)
    df["luz_min_ratio"]    = df[luz_cols].min(axis=1)  / (df["luz_mean"]  + 1e-6)
    df["agua_min_ratio"]   = df[agua_cols].min(axis=1) / (df["agua_mean"] + 1e-6)
    df["luz_ultimo_ratio"] = df["luz_dia7"]  / (df["luz_mean"]  + 1e-6)
    df["agua_ultimo_ratio"]= df["agua_dia7"] / (df["agua_mean"] + 1e-6)
    return df


# =============================================================================
# 3. ENTRENAMIENTO DE LA RED NEURONAL
# =============================================================================

def train_neural_network(df: pd.DataFrame) -> tuple:
    """
    Entrena un MLP (Perceptrón Multicapa) para clasificación binaria de anomalías.
    """
    df = add_statistical_features(df)

    feature_cols = (
        [f"luz_dia{i}"  for i in range(1, 8)] +
        [f"agua_dia{i}" for i in range(1, 8)] +
        ["luz_std", "agua_std", "luz_min_ratio", "agua_min_ratio",
         "luz_ultimo_ratio", "agua_ultimo_ratio"]
    )

    X = df[feature_cols].values
    y = df["anomalia"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # MLP con arquitectura 22→64→32→1
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,            # Regularización L2
        batch_size=32,
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    model.fit(X_train_sc, y_train)

    return model, scaler, X_test_sc, y_test, feature_cols


# =============================================================================
# 4. EVALUACIÓN
# =============================================================================

def evaluate_model(model, X_test_sc, y_test):
    """Calcula métricas de clasificación y curva de pérdida."""
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    print("=" * 55)
    print("  VITABEATS — Red Neuronal (MLP) · Resultados")
    print("=" * 55)
    print(f"\n  Precisión (Accuracy):  {accuracy_score(y_test, y_pred):.2%}")
    print(f"  AUC-ROC:               {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n  Épocas de entrenamiento: {model.n_iter_}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Normal', 'Anomalía'])}")


def plot_training_loss(model):
    """Visualiza la curva de pérdida durante el entrenamiento."""
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(model.loss_curve_, color="#00d4ff", linewidth=2, label="Pérdida entrenamiento")
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        ax_r = ax.twinx()
        ax_r.plot(model.validation_scores_, color="#7fff72", linewidth=2,
                  linestyle="--", label="Score validación")
        ax_r.set_ylabel("Score validación", fontsize=9, color="#7fff72")
        ax_r.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("Épocas", fontsize=10)
    ax.set_ylabel("Pérdida (loss)", fontsize=10, color="#00d4ff")
    ax.set_title("VitaBeats — Curva de entrenamiento de la Red Neuronal", fontsize=11, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig("neural_network_loss.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Gráfica guardada: neural_network_loss.png")


# =============================================================================
# 5. VISUALIZACIÓN DE UNA SERIE TEMPORAL CON ANOMALÍA
# =============================================================================

def plot_sensor_series(luz_vals: list, agua_vals: list, anomalia_pred: bool, score: float):
    """Visualiza los datos de sensores de 7 días con la anomalía marcada."""
    dias = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

    colores_luz  = ["#ff3366" if (i == 6 and anomalia_pred) else "#ffcc00" for i in range(7)]
    colores_agua = ["#ff3366" if (i == 6 and anomalia_pred) else "#00d4ff" for i in range(7)]

    axes[0].bar(dias, luz_vals, color=colores_luz, edgecolor="white", linewidth=0.5)
    axes[0].set_title("💡 Uso de luz (h/día)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Horas", fontsize=9)
    axes[0].set_facecolor("#f8f9fa")

    axes[1].bar(dias, agua_vals, color=colores_agua, edgecolor="white", linewidth=0.5)
    axes[1].set_title("💧 Uso de agua (L/día)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Litros", fontsize=9)
    axes[1].set_facecolor("#f8f9fa")

    estado = f"⚠️  ANOMALÍA DETECTADA (score: {score:.0%})" if anomalia_pred else f"✅ Patrón normal (score: {score:.0%})"
    fig.suptitle(f"VitaBeats — Serie temporal de sensores\n{estado}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("sensor_series.png", dpi=150, bbox_inches="tight")
    plt.show()


# =============================================================================
# 6. PREDICCIÓN PARA CASO NUEVO
# =============================================================================

def predict_sensors(model, scaler, luz_vals: list, agua_vals: list) -> tuple:
    """Analiza 7 días de datos de sensores y devuelve score de anomalía."""
    assert len(luz_vals) == 7 and len(agua_vals) == 7, "Se necesitan exactamente 7 días de datos"

    row = {f"luz_dia{i+1}": luz_vals[i] for i in range(7)}
    row.update({f"agua_dia{i+1}": agua_vals[i] for i in range(7)})

    df_temp = add_statistical_features(pd.DataFrame([row]))
    stat_cols = ["luz_std", "agua_std", "luz_min_ratio", "agua_min_ratio",
                 "luz_ultimo_ratio", "agua_ultimo_ratio"]
    feat_cols = [f"luz_dia{i}" for i in range(1,8)] + [f"agua_dia{i}" for i in range(1,8)] + stat_cols

    X_new = df_temp[feat_cols].values
    X_sc  = scaler.transform(X_new)
    score = float(model.predict_proba(X_sc)[0, 1])
    anomalia = score >= 0.5

    print(f"\n  🧠 Análisis de sensores (7 días):")
    print(f"     Luz:  {luz_vals}")
    print(f"     Agua: {agua_vals}")
    print(f"\n  🎯 Score de anomalía: {score:.2%}")
    print(f"  {'⚠️  ANOMALÍA DETECTADA' if anomalia else '✅ Patrón normal'}")

    nivel = "CRÍTICA" if score > 0.75 else "MODERADA" if score > 0.5 else "SIN ALERTA"
    print(f"  🔔 Nivel de alerta: {nivel}")

    plot_sensor_series(luz_vals, agua_vals, anomalia, score)
    return score, anomalia


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n  🧠 Iniciando VitaBeats — Módulo Red Neuronal...\n")

    df = generate_sensor_dataset(n_samples=800)
    print(f"  Dataset generado: {len(df)} registros")
    print(f"  Distribución anomalías: {df['anomalia'].value_counts().to_dict()}\n")

    model, scaler, X_test_sc, y_test, features = train_neural_network(df)
    evaluate_model(model, X_test_sc, y_test)
    plot_training_loss(model)

    # Caso ejemplo: datos de María Rodríguez con anomalía en domingo
    luz_maria  = [9.0, 10.0, 8.0, 11.0, 9.0, 10.0, 1.0]
    agua_maria = [45.0, 48.0, 42.0, 50.0, 46.0, 44.0, 9.0]
    predict_sensors(model, scaler, luz_maria, agua_maria)
