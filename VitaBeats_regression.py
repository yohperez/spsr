"""
VitaBeats — Módulo 2: Regresión Lineal
========================================
Predice cuántos días faltan hasta que el estado de ánimo o la salud
de la persona mayor decaiga a un nivel crítico (bienestar < 4/10)
si no recibe interacción social adicional.

Variable objetivo (target):
    dias_hasta_declive : Días hasta bienestar crítico (variable continua)

Features:
    - bienestar_actual      : Puntuación actual de bienestar (0-10)
    - dias_sin_social       : Días consecutivos sin interacción social
    - tendencia_codigo      : 0=Mejorando, 1=Estable, 2=Declinando
    - edad                  : Edad de la persona (años)
    - condicion_salud_cod   : 0=Ninguna, 1=Leve, 2=Moderada, 3=Severa
    - interacciones_mes_avg : Promedio de interacciones sociales por semana (último mes)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# 1. GENERACIÓN DE DATOS SINTÉTICOS
# =============================================================================

def generate_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """
    Genera dataset sintético para regresión.
    La variable objetivo se calcula con la fórmula:
        dias = (bienestar_actual - 4) / tasa_declive
    donde tasa_declive depende de tendencia, edad y condición de salud.
    """
    np.random.seed(random_state)

    bienestar   = np.round(np.random.uniform(4.1, 10.0, n_samples), 1)
    dias_social = np.random.randint(0, 21, n_samples)
    tendencia   = np.random.randint(0, 3, n_samples)   # 0=mejor, 1=estable, 2=declive
    edad        = np.random.randint(60, 95, n_samples)
    salud       = np.random.randint(0, 4,  n_samples)
    interacc    = np.round(np.random.uniform(0, 15, n_samples), 1)

    # Cálculo de la tasa de declive
    tasa_base = np.where(tendencia == 2, 0.30, np.where(tendencia == 1, 0.15, 0.05))
    factor_edad   = np.where(edad > 75, 1.2, 1.0)
    factor_salud  = np.select([salud == 3, salud == 2, salud == 1], [1.4, 1.2, 1.1], default=1.0)
    factor_social = 1.0 + dias_social * 0.02
    factor_interacc = np.where(interacc < 3, 1.15, 1.0)

    tasa = tasa_base * factor_edad * factor_salud * factor_social * factor_interacc
    dias = np.maximum(1, (bienestar - 4.0) / tasa)
    # Añadir ruido gaussiano realista
    dias = np.round(dias + np.random.normal(0, 1.5, n_samples), 1)
    dias = np.maximum(1, dias)

    df = pd.DataFrame({
        "bienestar_actual":      bienestar,
        "dias_sin_social":       dias_social,
        "tendencia_codigo":      tendencia,
        "edad":                  edad,
        "condicion_salud_cod":   salud,
        "interacciones_mes_avg": interacc,
        "dias_hasta_declive":    dias,
    })
    return df


# =============================================================================
# 2. ENTRENAMIENTO
# =============================================================================

def train_regression(df: pd.DataFrame) -> tuple:
    """
    Entrena un modelo de regresión Ridge (regularización L2 para mayor robustez).
    """
    features = ["bienestar_actual", "dias_sin_social", "tendencia_codigo",
                "edad", "condicion_salud_cod", "interacciones_mes_avg"]
    target   = "dias_hasta_declive"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = Ridge(alpha=1.0)
    model.fit(X_train_sc, y_train)

    return model, scaler, X_train, X_test, y_train, y_test, features


# =============================================================================
# 3. EVALUACIÓN
# =============================================================================

def evaluate_model(model, scaler, X_test, y_test, features: list):
    """Calcula y muestra métricas de regresión."""
    y_pred = model.predict(scaler.transform(X_test))
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("=" * 55)
    print("  VITABEATS — Regresión · Resultados")
    print("=" * 55)
    print(f"\n  MAE  (Error Absoluto Medio):  {mae:.2f} días")
    print(f"  RMSE (Raíz Error Cuadrático): {rmse:.2f} días")
    print(f"  R²   (Coeficiente det.):       {r2:.4f}")
    print(f"\n  Coeficientes del modelo:")
    for f, c in sorted(zip(features, model.coef_), key=lambda x: abs(x[1]), reverse=True):
        direction = "↑" if c > 0 else "↓"
        print(f"     {direction} {f:<30} {c:+.3f}")

    return y_pred


def plot_residuals(y_test, y_pred):
    """Gráfica de valores reales vs predichos y distribución de residuos."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Real vs Predicho
    axes[0].scatter(y_test, y_pred, alpha=0.5, color="#00d4ff", edgecolors="none", s=25)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1.5, label="Predicción perfecta")
    axes[0].set_xlabel("Días reales", fontsize=10)
    axes[0].set_ylabel("Días predichos", fontsize=10)
    axes[0].set_title("Real vs Predicho", fontsize=11, fontweight="bold")
    axes[0].legend()
    axes[0].set_facecolor("#f8f9fa")

    # Distribución de residuos
    residuos = np.array(y_test) - y_pred
    axes[1].hist(residuos, bins=30, color="#7fff72", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residuo (días)", fontsize=10)
    axes[1].set_ylabel("Frecuencia", fontsize=10)
    axes[1].set_title("Distribución de residuos", fontsize=11, fontweight="bold")
    axes[1].set_facecolor("#f8f9fa")

    fig.suptitle("VitaBeats — Regresión Lineal", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("regression_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  Gráfica guardada: regression_results.png")


# =============================================================================
# 4. PREDICCIÓN PARA CASO NUEVO
# =============================================================================

def predict_case(model, scaler, features: list, case: dict) -> float:
    """Predice los días hasta declive para un caso concreto."""
    X_new = pd.DataFrame([case])[features]
    dias  = float(model.predict(scaler.transform(X_new))[0])
    dias  = max(1, round(dias, 1))
    urgencia = "INMEDIATA 🔴" if dias <= 3 else "ESTA SEMANA 🟡" if dias <= 7 else "ESTE MES 🟢"

    print(f"\n  📋 Caso analizado:")
    for k, v in case.items():
        print(f"     {k}: {v}")
    print(f"\n  🎯 Predicción: {dias} días hasta nivel crítico")
    print(f"  ⏱  Urgencia:   {urgencia}")
    return dias


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("\n  📈 Iniciando VitaBeats — Módulo Regresión...\n")

    df = generate_dataset(n_samples=600)
    print(f"  Dataset generado: {len(df)} registros")
    print(f"  Target (días): media={df['dias_hasta_declive'].mean():.1f}  "
          f"min={df['dias_hasta_declive'].min():.1f}  max={df['dias_hasta_declive'].max():.1f}\n")

    model, scaler, X_train, X_test, y_train, y_test, features = train_regression(df)
    y_pred = evaluate_model(model, scaler, X_test, y_test, features)
    plot_residuals(y_test, y_pred)

    # Caso ejemplo: María Rodríguez
    caso_maria = {
        "bienestar_actual":      6.2,
        "dias_sin_social":       3,
        "tendencia_codigo":      2,   # Declinando
        "edad":                  78,
        "condicion_salud_cod":   1,   # Leve
        "interacciones_mes_avg": 4.0,
    }
    predict_case(model, scaler, features, caso_maria)
