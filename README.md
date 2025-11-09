# Empathy-LLM-Dataset-Pipeline

Pipeline completo para el desarrollo de un asistente de IA conversacional con capacidades de empatía y detección de riesgos en salud mental, desplegable en dispositivos móviles Android.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)
[![React Native](https://img.shields.io/badge/React_Native-0.76.6-blue.svg)](https://reactnative.dev/)

---

## 📋 Descripción

Este repositorio contiene el **pipeline completo de desarrollo** de un asistente de IA conversacional para apoyo emocional en salud mental, desarrollado como Trabajo de Fin de Máster (TFM) en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU).

El proyecto abarca **tres componentes principales**:

1. **Pipeline de generación de dataset** (9 scripts Python): Procesamiento de 20,132 ejemplos de diálogos empáticos en español peruano
2. **Pipeline de entrenamiento y optimización** (8 scripts Python): Fine-tuning con DoRA, cuantización Q4_K_M, evaluación multi-LLM
3. **Aplicación móvil de despliegue** (React Native): Interfaz de usuario para ejecutar el modelo on-device en Android

---

## 🎯 Objetivo del Proyecto

Desarrollar un asistente de IA conversacional **100% offline y privado** para apoyo emocional en dispositivos móviles, con un modelo de lenguaje de **1B de parámetros** optimizado mediante:

- **Fine-tuning con DoRA** (Weight-Decomposed Low-Rank Adaptation)
- **Cuantización Q4_K_M** (700MB, ejecutable en dispositivos de gama media)
- **RAG culturalmente adaptado** (modismos peruanos, criterios psicológicos)
- **Evaluación multi-LLM** (GPT-4.1-mini + Gemini 1.5 Pro como evaluadores)

---

## 🗂️ Estructura del Repositorio

```
empathy-llm-dataset-pipeline/
│
├── scripts/
│   ├── 1_classify_alexander_street.py
│   ├── 2_segment_academic_texts.py
│   ├── 3_classify_chunks_by_theme.py
│   ├── 3.5_download_daic_woz_transcripts.py
│   ├── 4_process_daic_woz.py
│   ├── 5_merge_knowledge_bases.py
│   ├── 6_rag_dataset_generator.py
│   ├── 7_generate_multimodal_examples.py
│   ├── 8_smote_implementation.py
│   ├── 9_generate_cultural_dataset.py
│   │
│   ├── ENTRENAMIENTO/
│   │   ├── 1_train_dora_empathy_smote_fixed.py
│   │   ├── 2_test_empathy_model.py
│   │   ├── 3_fusion_empathy_model.py
│   │   ├── 4_convert_empathy_to_gguf_f16.py
│   │   ├── 5_quantize_empathy_q4_from_f6.py
│   │   └── 6_evaluate_model_with_rag.py
│   │
│   ├── test_model/
│   │   ├── 6_evaluate_model_with_rag.py
│   │   └── 7_calculate_additional_metrics.py
│   │
│   └── EmocionalApp/                    # Aplicación móvil React Native
│       ├── android/                     # Configuración Android
│       ├── ios/                         # Configuración iOS
│       ├── src/                         # Código fuente de la app
│       ├── App.tsx                      # Componente principal
│       ├── package.json
│       ├── README.md                    # Guía de instalación de la app
│       └── GUIA_GENERAR_APK.md         # Guía para generar APK
│
├── docs/
│   ├── metodologia.md
│   ├── apendice_b_scripts.pdf
│   └── pipeline_diagram.png
│
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

## 🚀 Pipeline Completo del Proyecto

### **Fase I: Generación de Dataset** (9 scripts)

Procesamiento de dos fuentes principales de datos para crear un dataset de 20,132 ejemplos:

#### Rama A: Alexander Street (Base de Conocimiento Académico)

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| **Script 1** | Clasificar documentos en diálogos vs. textos académicos | 1,330 archivos `.txt` | 25 diálogos + 1,305 textos |
| **Script 2** | Segmentar textos en chunks de ~500 palabras | 1,305 textos | ~26,000 chunks |
| **Script 3** | Clasificar chunks por tema clínico | ~26,000 chunks | Chunks etiquetados |

#### Rama B: DAIC-WOZ (Entrevistas Clínicas Reales)

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| **Script 3.5** | Descargar transcripciones desde servidor oficial | 140 IDs | 140 archivos CSV |
| **Script 4** | Traducir al español peruano y formatear | 140 transcripciones | 140 diálogos JSON |

#### Fusión y Generación Sintética

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| **Script 5** | Fusionar ambas fuentes en base de conocimiento | Chunks + diálogos | `knowledge_base.json` |
| **Script 6** | Generar diálogos sintéticos con RAG | Base de conocimiento | 15,000 diálogos |
| **Script 7** | Generar ejemplos multimodales | Dataset de emojis | 1,000 ejemplos |
| **Script 8** | Balancear clases con SMOTE | 16,000 ejemplos | **20,132 ejemplos** |
| **Script 9** | Generar dataset culturalmente adaptado | Modismos peruanos | Dataset final JSONL |

**Tiempo total estimado**: 16-22 horas | **Costo API**: $75-120 USD

---

### **Fase II: Entrenamiento y Optimización** (8 scripts)

Fine-tuning del modelo Llama 3.2 1B y optimización para despliegue móvil:

| Script | Propósito | Input | Output | Tiempo |
|--------|-----------|-------|--------|--------|
| **Script 1** | Fine-tuning con DoRA | Dataset JSONL (20,132) | Modelo adaptado | 3-4 horas |
| **Script 2** | Prueba del modelo | Modelo + casos de prueba | Métricas de calidad | 10-15 min |
| **Script 3** | Fusión de adaptadores | Modelo base + DoRA | Modelo fusionado FP16 | 5 min |
| **Script 4** | Conversión a GGUF FP16 | Modelo PyTorch | GGUF FP16 (~2.6GB) | 5 min |
| **Script 5** | Cuantización Q4_K_M | GGUF FP16 | GGUF Q4_K_M (~700MB) | 5 min |
| **Script 6** | Evaluación con RAG | Modelo + 82 casos | Reporte JSON | 20-25 min |
| **Script 7** | Métricas adicionales | Reporte JSON | Métricas cuantitativas | 10 min |
| **Script 8** | Análisis de resultados | Métricas | Gráficos + estadísticas | 5 min |

**Tiempo total estimado**: 4-5 horas | **Hardware**: GPU 12+ GB VRAM (RTX 3080 Ti, RTX 4070, A5000)

---

### **Fase III: Despliegue Móvil** (Aplicación React Native)

Aplicación móvil Android para ejecutar el modelo cuantizado de forma local y privada.

#### **Características principales**:

- ✅ **Procesamiento 100% local**: Sin conexión a internet requerida
- ✅ **Privacidad absoluta**: Datos nunca salen del dispositivo
- ✅ **Modelo optimizado**: GGUF Q4_K_M (700MB) ejecutable en dispositivos de gama media
- ✅ **Interfaz conversacional**: Chat empático con detección de riesgos
- ✅ **Adaptación cultural**: Modismos peruanos integrados

#### **Requisitos del dispositivo**:

- Android 6.0 (API 23) o superior
- 6GB RAM mínimo (recomendado 8GB)
- 2GB espacio de almacenamiento
- Procesador ARM 64-bit (Snapdragon 660 o superior)

#### **Instalación y uso**:

Ver documentación completa en [`scripts/EmocionalApp/README.md`](scripts/EmocionalApp/README.md) y [`scripts/EmocionalApp/GUIA_GENERAR_APK.md`](scripts/EmocionalApp/GUIA_GENERAR_APK.md).

**Nota**: La aplicación móvil es el **vehículo de despliegue** del modelo de IA. El foco de este proyecto es el **desarrollo del modelo de lenguaje** (fine-tuning, cuantización, evaluación), no la ingeniería de software móvil.

---

## 📊 Dataset Final

### Composición

- **Alexander Street:** 26,000 chunks + 25 diálogos
- **DAIC-WOZ:** 140 diálogos traducidos
- **RAG:** 15,000 diálogos sintéticos
- **Multimodal:** 1,000 ejemplos
- **SMOTE:** +4,132 ejemplos sintéticos

**Total:** **20,132 ejemplos** en formato JSONL

### Distribución de Clases de Riesgo

| Clase | Ejemplos | Porcentaje |
|-------|----------|------------|
| `NO_CRISIS` | ~8,000 | 40% |
| `LOW_DISTRESS` | ~6,000 | 30% |
| `MODERATE` | ~4,000 | 20% |
| `HIGH_SUICIDE_RISK` | ~2,132 | 10% |

---

## 🛠️ Tecnologías Utilizadas

### Pipeline de Datos y Entrenamiento

- **Python 3.11**
- **PyTorch** (fine-tuning)
- **Transformers** (Hugging Face)
- **LangChain** (segmentación de texto)
- **scikit-learn** (TF-IDF, SMOTE)
- **NumPy, pandas** (procesamiento de datos)

### Optimización y Cuantización

- **DoRA** (Weight-Decomposed Low-Rank Adaptation)
- **llama.cpp** (inferencia optimizada en CPU)
- **GGUF** (formato de cuantización)

### Evaluación

- **OpenAI API** (GPT-4.1-mini como evaluador)
- **Google Gemini API** (Gemini 1.5 Pro como evaluador)
- **RAG** (Retrieval-Augmented Generation)

### Aplicación Móvil

- **React Native 0.76.6**
- **TypeScript**
- **llama.cpp** (motor de inferencia)
- **React Native GGUF** (integración del modelo)

### Datasets Fuente

- **Alexander Street Press** - Counseling and Psychotherapy Transcripts
- **DAIC-WOZ** - Distress Analysis Interview Corpus (USC)
- **Emoji Sentiment Ranking v1.0**

---

## 📦 Instalación

### Requisitos Previos

- Python 3.11+
- Cuenta de OpenAI con API key
- Cuenta de Google Cloud con Gemini API key
- Acceso a los datasets fuente (Alexander Street y DAIC-WOZ)
- GPU con 12+ GB VRAM (para entrenamiento)

### Pasos de Instalación

1. **Clonar el repositorio:**

```bash
git clone https://github.com/paul0610/empathy-llm-dataset-pipeline.git
cd empathy-llm-dataset-pipeline
```

2. **Crear entorno virtual:**

```bash
python3.11 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias:**

```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**

```bash
cp .env.example .env
# Editar .env y añadir:
# - OPENAI_API_KEY
# - GOOGLE_API_KEY (para Gemini)
# - Rutas a datasets fuente
```

---

## 🔧 Uso

### Ejecución Secuencial del Pipeline Completo

#### **Paso 1: Generar Dataset** (16-22 horas, $75-120 USD)

```bash
# Fase I - Rama A: Alexander Street
python scripts/1_classify_alexander_street.py
python scripts/2_segment_academic_texts.py
python scripts/3_classify_chunks_by_theme.py

# Fase I - Rama B: DAIC-WOZ
python scripts/3.5_download_daic_woz_transcripts.py
python scripts/4_process_daic_woz.py

# Fase II: Fusión y Generación
python scripts/5_merge_knowledge_bases.py
python scripts/6_rag_dataset_generator.py
python scripts/7_generate_multimodal_examples.py

# Fase III: Balanceo
python scripts/8_smote_implementation.py
python scripts/9_generate_cultural_dataset.py
```

#### **Paso 2: Entrenar y Optimizar Modelo** (4-5 horas, GPU 12+ GB)

```bash
cd scripts/ENTRENAMIENTO

# 1. Fine-tuning con DoRA (3-4 horas)
python 1_train_dora_empathy_smote_fixed.py

# 2. Prueba del modelo (10-15 min)
python 2_test_empathy_model.py

# 3. Fusión de adaptadores (5 min)
python 3_fusion_empathy_model.py

# 4. Conversión a GGUF FP16 (5 min)
python 4_convert_empathy_to_gguf_f16.py

# 5. Cuantización Q4_K_M (5 min)
python 5_quantize_empathy_q4_from_f6.py

# 6. Evaluación con RAG (20-25 min, $2-3 USD)
python 6_evaluate_model_with_rag.py
```

#### **Paso 3: Evaluación y Métricas** (15 min, $0.50 USD)

```bash
cd scripts/test_model

# 7. Evaluación con RAG (si no se hizo en Paso 2)
python 6_evaluate_model_with_rag.py

# 8. Métricas adicionales
python 7_calculate_additional_metrics.py
```

#### **Paso 4: Desplegar en Aplicación Móvil** (Opcional)

Ver [`scripts/EmocionalApp/README.md`](scripts/EmocionalApp/README.md) para instrucciones detalladas.

---

## 💰 Costos Estimados

### Costos de API (OpenAI + Google)

| Fase | Scripts | Tiempo | Costo |
|------|---------|--------|-------|
| **Generación de dataset** | Scripts 3, 4, 6, 7 | 16-22 horas | $75-120 |
| **Evaluación multi-LLM** | Script 6 (ENTRENAMIENTO) | 20-25 min | $2-3 |
| **Métricas adicionales** | Script 7 (test_model) | 10 min | $0.50 |
| **TOTAL** | - | **~20 horas** | **$77-123** |

### Costos de Hardware

- **GPU para entrenamiento**: RTX 3080 Ti (12GB VRAM) o superior
- **Tiempo de entrenamiento**: 3-4 horas
- **Costo eléctrico estimado**: $1-2 USD (dependiendo de tarifa local)

---

## 📄 Formato del Dataset

### Estructura JSONL

Cada línea del archivo `.jsonl` es un objeto JSON:

```json
{
  "dialog_id": "rag-empathy-12345",
  "turns": [
    {"role": "user", "text": "Me siento muy ansioso por los exámenes..."},
    {"role": "assistant", "text": "Entiendo que te sientas así..."}
  ],
  "labels": {
    "risk_class": "LOW_DISTRESS",
    "risk_signals": ["ansiedad_académica"],
    "category": "empathy_training",
    "source": "rag_generated"
  },
  "meta": {
    "language": "es-PE",
    "domain": ["academic"],
    "generation_method": "rag_tfidf_gpt4.1mini"
  }
}
```

### Clases de Riesgo

- `NO_CRISIS`: Sin señales de riesgo
- `LOW_DISTRESS`: Malestar leve
- `MODERATE`: Malestar moderado
- `HIGH_SUICIDE_RISK`: Riesgo alto/ideación suicida

---

## 🔬 Metodología RAG

El sistema de Generación Aumentada por Recuperación (RAG) implementado utiliza:

1. **Indexación:** TF-IDF con 5,000 features y n-gramas (1,2)
2. **Recuperación:** Similitud coseno para seleccionar top-k chunks
3. **Few-shot Prompting:** 1-2 diálogos de ejemplo en el prompt
4. **Generación:** gpt-4.1-mini con temperatura 0.8
5. **Validación:** Parseo y verificación de formato JSON

---

## 📈 Resultados del Modelo

### Métricas de Evaluación Clínica (82 casos DAIC-WOZ)

| Dimensión | Baseline | Fine-tuned | Mejora |
|-----------|----------|------------|--------|
| **Empatía** | 3.13 | 2.76 | -11.8% |
| **Reconocimiento Emocional** | 2.87 | 3.42 | +19.2% |
| **Detección de Crisis** | 1.29 | 3.79 | **+193.8%** |
| **Respuesta Apropiada** | 2.45 | 3.18 | +29.8% |
| **Calidad General** | 2.61 | 3.21 | +23.0% |
| **Promedio Global** | 2.47 | 3.27 | **+50.8%** |

### Métricas de Clasificación de Riesgo

| Métrica | Valor |
|---------|-------|
| **Recall (Sensitivity)** | >70% |
| **F1-Score** | 0.74 |
| **Precision** | ~75-80% |
| **Acuerdo inter-evaluador** | >82% |

### Comparación con Estado del Arte

| Sistema | Precisión | Privacidad | Offline | Adaptación Cultural |
|---------|-----------|------------|---------|---------------------|
| **Woebot** | 60-70% | Nube | No | Genérica (inglés) |
| **Wysa** | 65-75% | Nube | No | Genérica (multiidioma) |
| **Este trabajo** | **>70%** | **Absoluta** | **Sí** | **Específica Perú** |

---

## 📚 Documentación Adicional

- **Metodología completa:** Ver `docs/metodologia.md`
- **Apéndice de scripts:** Ver `docs/apendice_b_scripts.pdf`
- **Diagrama del pipeline:** Ver `docs/pipeline_diagram.png`
- **Guía de la app móvil:** Ver `scripts/EmocionalApp/README.md`
- **Guía para generar APK:** Ver `scripts/EmocionalApp/GUIA_GENERAR_APK.md`

---

## 🤝 Contribuciones

Este repositorio es parte de un Trabajo de Fin de Máster académico. Si deseas contribuir o tienes sugerencias:

1. Abre un **Issue** describiendo tu propuesta
2. Haz un **Fork** del repositorio
3. Crea una **Pull Request** con tus cambios

---

## 📖 Citación

Si utilizas este código o metodología en tu investigación, por favor cita:

```bibtex
@mastersthesis{Rojas2025empathy,
  title={Desarrollo de un Asistente de IA Conversacional con Capacidades de Empatía y Detección de Riesgos para Apoyo Emocional en Dispositivos Móviles},
  author={Paul Florencio Rojas Quispe},
  year={2025},
  school={Universidad Internacional de Valencia},
  type={Trabajo de Fin de Máster en Inteligencia Artificial}
}
```

---

## ⚖️ Licencia

Este proyecto está licenciado bajo la **MIT License**. Ver el archivo `LICENSE` para más detalles.

---

## 🙏 Agradecimientos

- **Alexander Street Press** por proporcionar acceso al corpus de transcripciones de psicoterapia
- **USC Institute for Creative Technologies** por el dataset DAIC-WOZ
- **Emoji Sentiment Ranking** por el dataset de sentimientos de emojis
- **OpenAI** por la API de GPT-4.1-mini
- **Google** por la API de Gemini 1.5 Pro
- **Universidad Internacional de Valencia (VIU)** por el apoyo académico
- **Dra. [Nombre de la psicóloga]** por la validación clínica del sistema

---

## 📧 Contacto

**Autor:** Paul Florencio Rojas Quispe  
**Email:** paulrojas0610@gmail.com  
**LinkedIn:** https://www.linkedin.com/in/paul-rojas-60bb35114/  
**Universidad:** Universidad Internacional de Valencia (VIU)  
**Programa:** Máster en Inteligencia Artificial

---

## 🔗 Enlaces Relevantes

- [Alexander Street Press](https://alexanderstreet.com/)
- [DAIC-WOZ Dataset](https://dcapswoz.ict.usc.edu/)
- [Emoji Sentiment Ranking](https://kt.ijs.si/data/Emoji_sentiment_ranking/)
- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Google Gemini API](https://ai.google.dev/)
- [LangChain Documentation](https://python.langchain.com/)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [React Native](https://reactnative.dev/)

---

## 📝 Notas Importantes

### Privacidad y Ética

- Este proyecto fue desarrollado con estricto apego a protocolos éticos de investigación en salud mental
- Los datos de DAIC-WOZ fueron utilizados bajo los términos de uso del dataset
- El modelo resultante está diseñado para **apoyo emocional**, no para diagnóstico clínico
- Se recomienda supervisión profesional en implementaciones reales
- La aplicación móvil garantiza **privacidad absoluta** mediante procesamiento 100% local

### Limitaciones

- El dataset está optimizado para español latinoamericano (variante peruana)
- Los diálogos sintéticos pueden contener sesgos del modelo generador (GPT-4.1-mini)
- La traducción de DAIC-WOZ puede haber perdido matices culturales
- El sistema **no reemplaza** la atención psicológica profesional
- La validación clínica se realizó con 82 casos; se requiere estudio longitudinal con más participantes

### Alcance del Proyecto

Este TFM se enfoca en el **desarrollo del modelo de IA** (generación de dataset, fine-tuning, cuantización, evaluación). La aplicación móvil React Native es el **vehículo de despliegue** que demuestra la viabilidad técnica del sistema, pero no es el foco principal del trabajo académico.

---

**Última actualización:** Noviembre 2025  
**Versión:** 2.0.0
