# Proyecto Propuesta: Análisis de URLs para Detección de Phishing  

## 📝 Descripción General  
Este proyecto corresponde a **un diseño de propuesta técnica y exploratoria**, cuyo objetivo principal es **analizar patrones y correlaciones en URLs** con el fin de aportar una **base para futuros modelos de detección de phishing**.  
La información utilizada proviene del conjunto de datos público de Kaggle, específicamente el dataset **hemanthpingali/phishing-url**, que contiene más de 247 000 registros con características sintácticas de URLs legítimas y maliciosas.  


---

## Contenido del Proyecto  

1. **Descarga y carga del dataset**  
   - Se realiza la descarga automática utilizando la librería `kagglehub`.  
   - Se lee el archivo `Dataset.csv` con `pandas`.

2. **Exploración inicial**  
   - Se examinan dimensiones y estructura del dataset (42 columnas).  
   - Se identifican características relacionadas a la composición sintáctica de URLs:  
     - Longitud de URL, número de puntos, caracteres especiales, entropía, etc.  

3. **Análisis de correlación**  
   - Se construye una matriz de correlación para determinar las variables más relevantes asociadas a la variable `Type` (0: legítimo, 1: phishing).  
   - Se observan correlaciones altas con características como:
     - `url_length`, `number_of_special_char_in_url`, `entropy_of_url`, entre otras.

4. **Selección preliminar de variables**  
   - Se eliminan columnas con baja significancia o alta redundancia.  
   - Se genera un nuevo `DataFrame` más compacto (`df1`) con las variables más relevantes para futuros modelos.

---

## 🧰 Herramientas utilizadas  

- 🐍 **Python**  
- 📦 Librerías:  
  - `pandas` (procesamiento de datos)  
  - `numpy` (operaciones numéricas)  
  - `seaborn` y `matplotlib` (visualizaciones)  
  - `kagglehub` (descarga automatizada del dataset)

---

## Propósito de la Propuesta  
- Explorar relaciones entre atributos de URLs y la probabilidad de phishing.  
- Identificar un subconjunto óptimo de variables para futuros modelos de machine learning.  
- Establecer una base estructurada para un pipeline de detección automatizada.  

 **Nota importante:** Este análisis **es una propuesta exploratoria**. 

---

## 📌 Próximos pasos sugeridos  

- Realizar un análisis de importancia de características usando modelos base.  
- Implementar un pipeline de preprocesamiento y entrenamiento supervisado.  
- Evaluar métricas de desempeño (precision, recall, F1-score, AUC).  
- Desarrollar un dashboard de monitoreo de URLs sospechosas.

---

## 📚 Fuente del Dataset  
- Kaggle: [https://www.kaggle.com/datasets/hemanthpingali/phishing-url](https://www.kaggle.com/datasets/hemanthpingali/phishing-url)
