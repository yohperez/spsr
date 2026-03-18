"""
VitaBeats — Módulo 1: Árbol de Decisión
========================================
Clasifica el nivel de riesgo de aislamiento social de personas mayores
en tres categorías: BAJO, MEDIO, ALTO.

Variables de entrada (features):
    - llamadas_semana    : Llamadas telefónicas por semana (int)
    - visitas_semana     : Visitas presenciales recibidas por semana (int)
    - dias_fuera_casa    : Días que salió del domicilio (0-7) (int)
    - actividades_social : Frecuencia de actividades sociales (0=ninguna, 1=ocasional, 2=regular)
    - bienestar_score    : Puntuación bienestar autopercibido (0-10) (float)
    - tiempo_solo_anios  : Años viviendo solo (float)

Variable objetivo (target):
    - riesgo_nivel       : 0=Bajo, 1=Medio, 2=Alto
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# =============================================================================

def generate_dataset(n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """
    Genera un dataset sintético realista para entrenamiento del árbol de decisión.
    Los patrones están basados en estudios clínicos sobre aislamiento social en mayores.
    """
    np.random.seed(random_state)

    data = {
        "llamadas_semana":    np.random.randint(0, 15, n_samples),
        "visitas_semana":     np.random.randint(0, 8,  n_samples),
        "dias_fuera_casa":    np.random.randint(0, 8,  n_samples),
        "actividades_social": np.random.randint(0, 3,  n_samples),
        "bienestar_score":    np.round(np.random.uniform(1.0, 10.0, n_samples), 1),
        "tiempo_solo_anios":  np.round(np.random.uniform(0.5, 20.0, n_samples), 1),
    }

    df = pd.DataFrame(data)

    # Regla de etiquetado basada en umbrales clínicos
    def label_risk(row):
        score = 0
        if row["llamadas_semana"] < 3:   score += 3
        elif row["llamadas_semana"] < 5: score += 1
        if row["visitas_semana"] == 0:   score += 3
        elif row["visitas_semana"] == 1: score += 1
        if row["dias_fuera_casa"] == 0:  score += 2
        elif row["dias_fuera_casa"] <= 1:score += 1
        if row["actividades_social"] == 0: score += 2
        if row["bienestar_score"] < 4:   score += 2
        elif row["bienestar_score"] < 6: score += 1
        if row["tiempo_solo_anios"] > 5: score += 1

        if score >= 7:   return 2  # ALTO
        elif score >= 3: return 1  # MEDIO
        else:            return 0  # BAJO

    df["riesgo_nivel"] = df.apply(label_risk, axis=1)
    return df


# =============================================================================
# 2. ENTRENAMIENTO DEL ÁRBOL DE DECISIÓN
# =============================================================================

def train_decision_tree(df: pd.DataFrame) -> tuple:
    """
    Entrena un árbol de decisión con los datos del dataset.
    Retorna el modelo entrenado, datos de test y predicciones.
    """
    features = ["llamadas_semana", "visitas_semana", "dias_fuera_casa",
                "actividades_social", "bienestar_score", "tiempo_solo_anios"]
    target = "riesgo_nivel"

    X = df[features]
    y = df[target]

    # División 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Árbol con profundidad máxima 5 para evitar sobreajuste
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, features


# =============================================================================
# 3. EVALUACIÓN Y VISUALIZACIÓN
# =============================================================================

def evaluate_model(model, X_test, y_test):
    """Evalúa el modelo y muestra métricas de rendimiento."""
    y_pred = model.predict(X_test)
    labels = ["Bajo", "Medio", "Alto"]

    print("=" * 55)
    print("  VITABEATS — Árbol de Decisión · Resultados")
    print("=" * 55)
    print(f"\n  Precisión global (Accuracy): {accuracy_score(y_test, y_pred):.2%}\n")
    print(classification_report(y_test, y_pred, target_names=labels))
    return y_pred


def plot_feature_importance(model, features: list):
    """Visualiza la importancia de cada variable en el árbol."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#00d4ff", "#ff6b35", "#7fff72", "#ffcc00", "#ff3366", "#a78bfa"]
    bars = ax.barh(
        [features[i] for i in indices],
        [importances[i] for i in indices],
        color=[colors[i % len(colors)] for i in range(len(indices))],
        edgecolor="none"
    )
    ax.set_xlabel("Importancia relativa", fontsize=10)
    ax.set_title("VitaBeats — Importancia de variables (Árbol de Decisión)", fontsize=12, fontweight="bold")
    ax.set_facecolor("#f8f9fa")
    fig.patch.set_facecolor("#ffffff")
    for bar in bars:
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.2f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig("decision_tree_features.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Gráfica guardada: decision_tree_features.png")


# =============================================================================
# 4. PREDICCIÓN PARA UN CASO NUEVO
# =============================================================================

def predict_case(model, features: list, case: dict) -> str:
    """Clasifica un caso nuevo y devuelve el nivel de riesgo."""
    X_new = pd.DataFrame([case])[features]
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    labels = {0: "BAJO 🟢", 1: "MEDIO 🟡", 2: "ALTO 🔴"}

    print("\n  📋 Caso analizado:")
    for k, v in case.items():
        print(f"     {k}: {v}")
    print(f"\n  🎯 Clasificación: {labels[pred]}")
    print(f"     Confianza: Bajo={proba[0]:.0%}  Medio={proba[1]:.0%}  Alto={proba[2]:.0%}")
    return labels[pred]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n  🌳 Iniciando VitaBeats — Módulo Árbol de Decisión...\n")

    # Generar datos
    df = generate_dataset(n_samples=500)
    print(f"  Dataset generado: {len(df)} registros")
    print(f"  Distribución: {df['riesgo_nivel'].value_counts().to_dict()}\n")

    # Entrenar
    model, X_train, X_test, y_train, y_test, features = train_decision_tree(df)

    # Evaluar
    evaluate_model(model, X_test, y_test)

    # Importancia de variables
    plot_feature_importance(model, features)

    # Reglas del árbol
    print("\n  📐 Reglas del árbol (primeros niveles):")
    print(export_text(model, feature_names=features, max_depth=3))

    # Caso de ejemplo: María Rodríguez
    caso_maria = {
        "llamadas_semana": 3,
        "visitas_semana": 1,
        "dias_fuera_casa": 2,
        "actividades_social": 1,
        "bienestar_score": 6.2,
        "tiempo_solo_anios": 3.0,
    }
    predict_case(model, features, caso_maria)
