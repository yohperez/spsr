# 💓 VitaBeat
### Sistema de Prevención de Soledad y Riesgo en Personas Mayores

> **Proyecto universitario** · Inteligencia Artificial Aplicada  
> Detección temprana de aislamiento social y anomalías de salud mediante algoritmos de ML/IA

---

## 📋 Descripción del Problema

En España, más del **27% de las personas mayores de 65 años viven solas**. El aislamiento social prolongado incrementa en un 29% el riesgo de enfermedad coronaria y en un 32% el riesgo de ictus. VitaBeats es un sistema de monitoreo inteligente que combina tres técnicas de IA para detectar, predecir y prevenir situaciones de riesgo antes de que ocurran.

---

## 🧠 Módulos de Inteligencia Artificial

| Módulo | Técnica | Objetivo |
|--------|---------|----------|
| 🌳 Árbol de Decisión | Clasificación supervisada | Clasificar riesgo de aislamiento: Bajo / Medio / Alto |
| 📈 Regresión Lineal | Regresión supervisada | Predecir días hasta declive del bienestar |
| 🧠 Red Neuronal (RNN/MLP) | Aprendizaje profundo | Detectar anomalías en series temporales de sensores |

---

## 📁 Estructura del Repositorio

```
vitabeats/
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias Python
├── src/
│   ├── decision_tree.py         # Módulo Árbol de Decisión
│   ├── regression.py            # Módulo Regresión Lineal
│   └── neural_network.py        # Módulo Red Neuronal (RNN/MLP)
├── notebooks/
│   └── VitaBeats_Analisis.ipynb # Notebook completo con análisis y visualizaciones
├── data/
│   └── sample_data.csv          # Dataset de ejemplo (datos sintéticos anonimizados)
└── vitabeats_dashboard.html     # Dashboard web interactivo con IA integrada
```

---

## ⚙️ Instalación

### Requisitos previos
- Python 3.9 o superior
- pip

### 1. Clonar el repositorio
```bash
git clone https://github.com/yohperez/spsr.git
cd spsr
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

---

## 🚀 Ejecución

### Ejecutar el árbol de decisión
```bash
python src/decision_tree.py
```

### Ejecutar la regresión
```bash
python src/regression.py
```

### Ejecutar la red neuronal
```bash
python src/neural_network.py
```

### Ejecutar el notebook completo
```bash
jupyter notebook notebooks/VitaBeats_Analisis.ipynb
```

### Abrir el dashboard web
Abre `vitabeats_dashboard.html` directamente en tu navegador (sin servidor necesario).

---

## 📊 Dataset

Los datos utilizados son **sintéticos y generados específicamente** para este proyecto, basados en patrones clínicos reales de estudios sobre aislamiento social en personas mayores.

**Variables principales:**
- `llamadas_semana`: Número de llamadas telefónicas recibidas/realizadas por semana
- `visitas_semana`: Número de visitas presenciales recibidas por semana
- `dias_fuera_casa`: Días que la persona salió del domicilio
- `bienestar_score`: Puntuación de bienestar autopercibido (0-10)
- `uso_luz_horas`: Horas de uso de luz eléctrica por día (sensor)
- `uso_agua_litros`: Litros de agua consumidos por día (sensor)
- `riesgo_nivel`: Variable objetivo → Bajo / Medio / Alto

---

## 👥 Equipo

| Miembro | Rol |
|---------|-----|
| [Nombre 1] | Árbol de Decisión + Dataset |
| [Nombre 2] | Regresión + Análisis de datos |
| [Nombre 3] | Red Neuronal + Dashboard web |
| [Nombre 4] | Documentación + Presentación |

---

## 📄 Licencia

MIT License — Proyecto académico sin fines comerciales.
# spsr
