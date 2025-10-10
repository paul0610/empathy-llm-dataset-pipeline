# Empathy-LLM-Dataset-Pipeline

Pipeline completo de procesamiento de datos para la generación de un dataset de entrenamiento de modelos de lenguaje con capacidades de empatía y detección de riesgos en salud mental.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![OpenAI API](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)

---

## 📋 Descripción

Este repositorio contiene los **9 scripts** desarrollados como parte del Trabajo de Fin de Máster (TFM) en Inteligencia Artificial de la Universidad Internacional de Valencia (VIU). El proyecto implementa un pipeline completo para la generación de un dataset de 20,132 ejemplos de diálogos empáticos en español latinoamericano (variante peruana), diseñado para entrenar modelos de lenguaje pequeños (Small Language Models) con capacidades de:

- **Empatía y validación emocional**
- **Detección de riesgos de salud mental**
- **Análisis multimodal textual** (emojis, patrones de escritura, análisis longitudinal)
- **Técnicas terapéuticas** (TCC, ACT, Entrevista Motivacional)

---

## 🎯 Objetivo del Proyecto

Desarrollar un asistente de IA conversacional 100% offline y privado para apoyo emocional en dispositivos móviles, con un modelo de lenguaje de 1B de parámetros optimizado mediante fine-tuning con DoRA (Weight-Decomposed Low-Rank Adaptation) y cuantización a 4 bits.

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
│   └── 8_smote_implementation.py
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

## 🚀 Pipeline de Procesamiento

El pipeline consta de **9 scripts** que procesan datos de dos fuentes principales:

### Fase I: Procesamiento de Fuentes de Datos

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

### Fase II: Fusión y Generación Sintética

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| **Script 5** | Fusionar ambas fuentes en base de conocimiento | Chunks + diálogos | `knowledge_base.json` |
| **Script 6** | Generar diálogos sintéticos con RAG | Base de conocimiento | 15,000 diálogos |
| **Script 7** | Generar ejemplos multimodales | Dataset de emojis | 1,000 ejemplos |

### Fase III: Balanceo

| Script | Propósito | Input | Output |
|--------|-----------|-------|--------|
| **Script 8** | Balancear clases con SMOTE | 16,000 ejemplos | **20,132 ejemplos** |

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

### Lenguajes y Frameworks

- **Python 3.11**
- **LangChain** (segmentación de texto)
- **scikit-learn** (TF-IDF, vectorización)
- **NumPy** (operaciones vectoriales)
- **pandas** (procesamiento de datos)

### APIs y Modelos

- **OpenAI API** (gpt-4.1-mini)
- **Emoji Sentiment Ranking v1.0** (dataset externo)

### Datasets Fuente

- **Alexander Street Press** - Counseling and Psychotherapy Transcripts
- **DAIC-WOZ** - Distress Analysis Interview Corpus (USC)

---

## 📦 Instalación

### Requisitos Previos

- Python 3.11+
- Cuenta de OpenAI con API key
- Acceso a los datasets fuente (Alexander Street y DAIC-WOZ)

### Pasos de Instalación

1. **Clonar el repositorio:**

```bash
git clone https://github.com/tu-usuario/empathy-llm-dataset-pipeline.git
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
# Editar .env y añadir tu OPENAI_API_KEY
```

---

## 🔧 Uso

### Ejecución Secuencial del Pipeline

Los scripts deben ejecutarse en orden:

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
```

### Configuración de Rutas

Cada script tiene variables de configuración al inicio del archivo. Ajusta las rutas según tu entorno:

```python
# Ejemplo en 1_classify_alexander_street.py
INPUT_DIR = "/ruta/a/alexander_street_data"
OUTPUT_DIR = "/ruta/a/salida"
```

---

## ⚙️ Parámetros Clave

### Script 2: Segmentación

```python
CHUNK_SIZE_WORDS = 500  # Tamaño de chunk en palabras
OVERLAP_WORDS = 50      # Overlap entre chunks
```

### Script 3: Clasificación Temática

```python
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0  # Determinista para clasificación
```

### Script 4: Traducción DAIC-WOZ

```python
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.3  # Baja para consistencia
```

### Script 6: Generación RAG

```python
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.8  # Creatividad moderada
TOP_K_CHUNKS = 2   # Chunks recuperados por consulta
```

### Script 8: SMOTE

```python
K_NEIGHBORS = 5      # Vecinos para interpolación
RANDOM_STATE = 42    # Reproducibilidad
```

---

## 💰 Costos Estimados

### Costos de API (OpenAI)

| Script | Tiempo Estimado | Costo Estimado |
|--------|-----------------|----------------|
| Script 3 | 2-3 horas | $5-10 |
| Script 4 | 2-3 horas | $10-15 |
| Script 6 | 8-12 horas | $50-80 |
| Script 7 | 2-3 horas | $10-15 |
| **Total** | **16-22 horas** | **$75-120** |

*Nota: Los costos son aproximados y dependen del pricing de OpenAI en el momento de ejecución.*

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

El sistema de Generación Aumentada por Recuperación (RAG) implementado en el Script 6 utiliza:

1. **Indexación:** TF-IDF con 5,000 features y n-gramas (1,2)
2. **Recuperación:** Similitud coseno para seleccionar top-k chunks
3. **Few-shot Prompting:** 1-2 diálogos de ejemplo en el prompt
4. **Generación:** gpt-4.1-mini con temperatura 0.8
5. **Validación:** Parseo y verificación de formato JSON

---

## 📚 Documentación Adicional

- **Metodología completa:** Ver `docs/metodologia.md`
- **Apéndice de scripts:** Ver `docs/apendice_b_scripts.pdf`
- **Diagrama del pipeline:** Ver `docs/pipeline_diagram.png`

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
  type={Trabajo de Fin de Máster}
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
- **OpenAI** por la API de gpt-4.1-mini
- **Universidad Internacional de Valencia (VIU)** por el apoyo académico

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
- [LangChain Documentation](https://python.langchain.com/)

---

## 📝 Notas Importantes

### Privacidad y Ética

- Este proyecto fue desarrollado con estricto apego a protocolos éticos de investigación en salud mental
- Los datos de DAIC-WOZ fueron utilizados bajo los términos de uso del dataset
- El modelo resultante está diseñado para **apoyo emocional**, no para diagnóstico clínico
- Se recomienda supervisión profesional en implementaciones reales

### Limitaciones

- El dataset está en español latinoamericano
- Los diálogos sintéticos pueden contener sesgos del modelo generador
- La traducción de DAIC-WOZ puede haber perdido matices culturales
- El sistema no reemplaza la atención psicológica profesional

---

**Última actualización:** Octubre 2025  
**Versión:** 1.0.0

