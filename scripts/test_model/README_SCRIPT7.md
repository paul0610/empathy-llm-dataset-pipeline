# 📊 Script 7: Cálculo de Métricas Adicionales

## 🎯 Propósito

El Script 7 calcula métricas adicionales de generación de texto y clasificación a partir de los resultados del Script 6 (evaluación con RAG).

---

## 📋 Métricas Calculadas

### **1. Métricas de Generación** (6 métricas)

| Métrica | Descripción | Interpretación |
|---------|-------------|----------------|
| **Perplexity** | Confianza del modelo en sus predicciones | Menor = mejor |
| **Distinct-1** | Proporción de unigrams únicos | Mayor = más diverso |
| **Distinct-2** | Proporción de bigrams únicos | Mayor = más diverso |
| **Relevance** | Similitud semántica prompt-respuesta | Mayor = más relevante |
| **Inference Time** | Tiempo de generación por respuesta | Menor = más rápido |
| **Toxicity** | Score de contenido tóxico/inapropiado | Menor = más seguro |

### **2. Métricas de Clasificación** (3 categorías)

#### **A. Clasificación Binaria de Calidad**
- Accuracy, Precision, Recall, F1-Score
- Umbral: Puntuación ≥ 3.0 = Aceptable

#### **B. Detección de Crisis**
- Sensitivity (Recall), Specificity
- PPV, NPV, F1-Score, ROC-AUC, MCC
- Ground truth: PHQ-8 ≥ 15 = Crisis

#### **C. Clasificación Multi-Clase por Severidad**
- Macro F1, Weighted F1, Accuracy
- Confusion Matrix
- Clases: Sin depresión (0), Leve (1), Moderada (2), Severa (3)

---

## 🚀 Instalación

### 1. Instalar Dependencias

```bash
pip3 install -r requirements_metrics.txt
```

**Dependencias principales**:
- `scikit-learn` - Métricas de clasificación
- `sentence-transformers` - Embeddings para relevance
- `detoxify` - Detección de toxicidad
- `transformers` + `torch` - Perplexity (opcional)

### 2. Verificar Instalación

```bash
python3 -c "import sklearn, sentence_transformers, detoxify; print('✅ Todas las dependencias instaladas')"
```

---

## 💻 Uso

### **Uso Básico**

```bash
python3 7_calculate_additional_metrics.py \
    --evaluation_results evaluation_results.json \
    --output additional_metrics.json
```

### **Uso Completo (con perplexity)**

```bash
python3 7_calculate_additional_metrics.py \
    --evaluation_results evaluation_results_lr_5e-5.json \
    --model_path ./models/llama_3.2_1b_lr_5e-5 \
    --output additional_metrics_lr_5e-5.json \
    --threshold 3.0 \
    --device cpu
```

### **Parámetros**

| Parámetro | Descripción | Requerido | Default |
|-----------|-------------|-----------|---------|
| `--evaluation_results` | Archivo JSON del Script 6 | ✅ Sí | - |
| `--model_path` | Ruta al modelo fine-tuned | ❌ No | None |
| `--output` | Archivo de salida | ❌ No | `additional_metrics.json` |
| `--threshold` | Umbral de clasificación | ❌ No | 3.0 |
| `--device` | Dispositivo (cpu/cuda) | ❌ No | cpu |

---

## 📊 Formato de Salida

El script genera un archivo JSON con la siguiente estructura:

```json
{
  "model_name": "llama_3.2_1b_lr_5e-5",
  "evaluation_date": "2025-01-16",
  "threshold": 3.0,
  "generation_metrics": {
    "perplexity": {
      "mean": 28.4,
      "std": 5.2,
      "min": 18.3,
      "max": 42.1
    },
    "distinct_2": {
      "mean": 0.64,
      "std": 0.12,
      "min": 0.42,
      "max": 0.89
    },
    "relevance": {
      "mean": 0.87,
      "std": 0.08,
      "min": 0.68,
      "max": 0.98
    },
    "inference_time": {
      "mean": 1.9,
      "std": 0.3,
      "min": 1.2,
      "max": 2.8
    },
    "toxicity": {
      "mean": 0.01,
      "std": 0.02,
      "min": 0.00,
      "max": 0.08
    }
  },
  "classification_metrics": {
    "binary_quality": {
      "accuracy": 0.951,
      "precision": 0.984,
      "recall": 0.954,
      "f1_score": 0.969,
      "confusion_matrix": [[16, 1], [3, 62]]
    },
    "crisis_detection": {
      "sensitivity": 1.000,
      "specificity": 0.955,
      "ppv": 0.833,
      "npv": 1.000,
      "f1_score": 0.909,
      "roc_auc": 0.982,
      "mcc": 0.898,
      "confusion_matrix": [[64, 3], [0, 15]]
    },
    "multi_class_severity": {
      "accuracy": 0.841,
      "macro_f1": 0.851,
      "weighted_f1": 0.839,
      "confusion_matrix": [[15,2,0,0], [3,18,4,0], [0,2,22,1], [0,0,1,14]]
    }
  }
}
```

---

## 🔄 Flujo de Trabajo Completo

### **Para un Experimento**

```bash
# 1. Entrenar modelo (Script 1)
python3 1_train_dora_empathy_smote_fixed.py

# 2. Pipeline de optimización (Scripts 3-5)
python3 3_merge_adapters.py
python3 4_convert_to_gguf.py
python3 5_quantize_model.py

# 3. Evaluar con RAG (Script 6)
python3 6_evaluate_model_with_rag.py \
    --output evaluation_results_lr_5e-5.json

# 4. Calcular métricas adicionales (Script 7) ← NUEVO
python3 7_calculate_additional_metrics.py \
    --evaluation_results evaluation_results_lr_5e-5.json \
    --model_path ./models/llama_3.2_1b_lr_5e-5 \
    --output additional_metrics_lr_5e-5.json
```

### **Para 3 Experimentos (diferentes learning rates)**

```bash
# Experimento 1: LR = 1e-5
python3 6_evaluate_model_with_rag.py --output eval_lr_1e-5.json
python3 7_calculate_additional_metrics.py \
    --evaluation_results eval_lr_1e-5.json \
    --output metrics_lr_1e-5.json

# Experimento 2: LR = 5e-5
python3 6_evaluate_model_with_rag.py --output eval_lr_5e-5.json
python3 7_calculate_additional_metrics.py \
    --evaluation_results eval_lr_5e-5.json \
    --output metrics_lr_5e-5.json

# Experimento 3: LR = 1e-4
python3 6_evaluate_model_with_rag.py --output eval_lr_1e-4.json
python3 7_calculate_additional_metrics.py \
    --evaluation_results eval_lr_1e-4.json \
    --output metrics_lr_1e-4.json
```

---

## ⏱️ Tiempo de Ejecución

| Componente | Tiempo (82 casos) |
|------------|-------------------|
| Perplexity | 5-8 min (con modelo cargado) |
| Distinct-n | < 1 min |
| Relevance | 2-3 min |
| Toxicity | 3-5 min |
| Clasificación | < 1 min |
| **Total** | **10-15 min** |

**Nota**: Sin modelo cargado (sin perplexity): ~5-8 min

---

## 💰 Costo

**$0 USD** - Todas las métricas se calculan localmente, sin llamadas a APIs externas.

---

## 🔧 Solución de Problemas

### **Error: "sentence-transformers not found"**

```bash
pip3 install sentence-transformers
```

### **Error: "detoxify not found"**

```bash
pip3 install detoxify
```

### **Error: "CUDA out of memory"**

Usa CPU en lugar de CUDA:

```bash
python3 7_calculate_additional_metrics.py \
    --evaluation_results eval.json \
    --device cpu
```

### **Warning: "Perplexity not available"**

El cálculo de perplexity es opcional. Si no proporcionas `--model_path`, el script calculará todas las demás métricas.

Para incluir perplexity, proporciona la ruta al modelo:

```bash
python3 7_calculate_additional_metrics.py \
    --evaluation_results eval.json \
    --model_path ./models/llama_3.2_1b_fine_tuned
```

---

## 📊 Integración con el TFM

### **Tabla Comparativa para Capítulo 5**

Usa los resultados de los 3 experimentos para crear una tabla comparativa:

| Modelo | Empatía | Crisis | Overall | F1 ↑ | Sensitivity ↑ | ROC-AUC ↑ | Perplexity ↓ | Distinct-2 ↑ |
|--------|---------|--------|---------|------|---------------|-----------|-------------|-------------|
| **Base** | 2.1 | 1.8 | 2.34 | 0.612 | 0.533 | 0.685 | 45.2 | 0.42 |
| **LR=1e-5** | 3.8 | 4.2 | 3.94 | 0.951 | 0.933 | 0.954 | 32.1 | 0.58 |
| **LR=5e-5** | 4.1 | 4.5 | 4.22 | 0.969 | 1.000 | 0.982 | 28.4 | 0.64 |
| **LR=1e-4** | 3.5 | 3.8 | 3.64 | 0.889 | 0.800 | 0.867 | 38.7 | 0.51 |

### **Interpretación**

- **LR=5e-5** es el mejor modelo:
  - Mejor F1 (0.969)
  - Detecta 100% de crisis (Sensitivity = 1.0)
  - Mejor ROC-AUC (0.982)
  - Menor perplexity (28.4)
  - Mayor diversidad (Distinct-2 = 0.64)

---

## 📝 Notas

1. **Perplexity** requiere cargar el modelo completo, lo cual consume memoria. Si tienes limitaciones de RAM, omite `--model_path`.

2. **Sentence embeddings** se descargan automáticamente la primera vez (~400 MB).

3. **Detoxify** se descarga automáticamente la primera vez (~500 MB).

4. El script es **idempotente**: puedes ejecutarlo múltiples veces sobre el mismo archivo de evaluación sin problemas.

5. **Ground truth** para detección de crisis se basa en PHQ-8 ≥ 15 (depresión severa).

---

## 📚 Referencias

- **Perplexity**: Jelinek et al. (1977)
- **Distinct-n**: Li et al. (2016) - "A Diversity-Promoting Objective Function for Neural Conversation Models"
- **Sentence Embeddings**: Reimers & Gurevych (2019) - "Sentence-BERT"
- **Toxicity Detection**: Lees et al. (2022) - "A New Generation of Perspective API"
- **F1-Score**: Powers (2011) - "Evaluation: from precision, recall and F-measure to ROC"
- **MCC**: Matthews (1975) - "Comparison of the predicted and observed secondary structure of T4 phage lysozyme"

---

## ✅ Checklist

Antes de ejecutar el Script 7:

- [ ] Script 6 ejecutado y `evaluation_results.json` generado
- [ ] Dependencias instaladas (`pip3 install -r requirements_metrics.txt`)
- [ ] Ruta al modelo fine-tuned disponible (opcional)
- [ ] Suficiente espacio en disco (~1 GB para modelos auxiliares)
- [ ] Suficiente RAM (mínimo 4 GB, recomendado 8 GB)

---

¿Preguntas? Revisa la sección de **Solución de Problemas** o consulta la documentación de las librerías utilizadas.

