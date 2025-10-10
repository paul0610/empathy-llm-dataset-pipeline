#!/usr/bin/env python3
"""
Script: 7_generate_multimodal_examples.py
Descripción: Genera 1,000 ejemplos con análisis de emojis, patrones de escritura
             y análisis longitudinal para entrenar capacidades multimodales textuales.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Descarga el dataset Emoji Sentiment Ranking v1.0
2. Genera 400 ejemplos de análisis de emojis
3. Genera 400 ejemplos de análisis de patrones de escritura
4. Genera 200 ejemplos de análisis longitudinal
5. Guarda los 1,000 ejemplos en formato JSONL
"""

import json
import random
from openai import OpenAI
import pandas as pd
import requests
import time
from typing import Dict, List
import os

# Configuración
client = OpenAI()  # API key ya configurada en variables de entorno

KNOWLEDGE_BASE_FILE = "/home/ubuntu/knowledge_base.json"
EMOJI_DATASET_URL = "https://kt.ijs.si/data/Emoji_sentiment_ranking/Emoji_Sentiment_Data_v1.0.csv"
EMOJI_DATASET_FILE = "/home/ubuntu/emoji_sentiment_ranking.csv"
OUTPUT_FILE = "/home/ubuntu/multimodal_examples.jsonl"
METADATA_FILE = "/home/ubuntu/multimodal_generation_metadata.json"

# Configuración del modelo
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.7

# Distribución de ejemplos
EXAMPLE_DISTRIBUTION = {
    'emoji_analysis': 400,
    'writing_patterns': 400,
    'longitudinal_analysis': 200
}


def download_emoji_dataset():
    """
    Descarga el dataset Emoji Sentiment Ranking v1.0.
    """
    if os.path.exists(EMOJI_DATASET_FILE):
        print(f"  ✓ Dataset de emojis ya existe: {EMOJI_DATASET_FILE}")
        return
    
    print(f"  → Descargando dataset de emojis desde {EMOJI_DATASET_URL}...")
    
    try:
        response = requests.get(EMOJI_DATASET_URL, timeout=30)
        
        if response.status_code == 200:
            with open(EMOJI_DATASET_FILE, 'wb') as f:
                f.write(response.content)
            print(f"  ✓ Dataset descargado exitosamente")
        else:
            print(f"  ✗ Error descargando dataset: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"  ✗ Error: {e}")


def load_emoji_dataset() -> pd.DataFrame:
    """
    Carga el dataset de emojis.
    
    Returns:
        DataFrame con los emojis y sus sentimientos
    """
    try:
        df = pd.read_csv(EMOJI_DATASET_FILE)
        print(f"  ✓ Dataset de emojis cargado: {len(df)} emojis")
        return df
    except Exception as e:
        print(f"  ✗ Error cargando dataset: {e}")
        return pd.DataFrame()


def generate_emoji_analysis_example(emoji_df: pd.DataFrame) -> Dict:
    """
    Genera un ejemplo de análisis de emojis.
    
    Args:
        emoji_df: DataFrame con emojis y sentimientos
    
    Returns:
        Ejemplo generado
    """
    # Seleccionar emojis aleatorios
    num_emojis = random.randint(1, 3)
    sample_emojis = emoji_df.sample(n=min(num_emojis, len(emoji_df)))
    
    emojis = sample_emojis['Emoji'].tolist() if 'Emoji' in sample_emojis.columns else ['😊', '😢']
    sentiments = sample_emojis['Sentiment'].tolist() if 'Sentiment' in sample_emojis.columns else [0.5, -0.3]
    
    # Construir prompt
    emoji_str = ' '.join(emojis)
    
    prompt = f"""Genera un diálogo corto (4-6 turnos) donde el usuario usa estos emojis: {emoji_str}

El asistente debe:
1. Reconocer el significado emocional de los emojis
2. Validar la emoción expresada
3. Responder de manera apropiada

FORMATO DE SALIDA (JSON):
{{
  "turns": [
    {{"role": "user", "text": "texto con emojis {emoji_str}"}},
    {{"role": "assistant", "text": "respuesta empática"}}
  ],
  "emoji_analysis": {{
    "emojis_detected": ["{emojis[0] if emojis else '😊'}"],
    "emotional_tone": "positivo|negativo|mixto",
    "interpretation": "breve interpretación"
  }},
  "risk_class": "NO_CRISIS|LOW_DISTRESS"
}}

GENERA EL DIÁLOGO:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Parsear JSON
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = generated_text[json_start:json_end]
            data = json.loads(json_text)
            
            example = {
                'dialog_id': f'multimodal-emoji-{random.randint(10000, 99999)}',
                'turns': data.get('turns', []),
                'labels': {
                    'risk_class': data.get('risk_class', 'NO_CRISIS'),
                    'emoji_analysis': data.get('emoji_analysis', {}),
                    'source': 'multimodal_generated'
                },
                'meta': {
                    'language': 'es-PE',
                    'domain': ['emoji_analysis'],
                    'multimodal_type': 'emoji'
                }
            }
            
            return example
        else:
            return None
    
    except Exception as e:
        print(f"    ✗ Error generando ejemplo de emoji: {e}")
        return None


def generate_writing_pattern_example() -> Dict:
    """
    Genera un ejemplo de análisis de patrones de escritura.
    
    Returns:
        Ejemplo generado
    """
    patterns = [
        "uso excesivo de puntos suspensivos...",
        "MAYÚSCULAS para expresar intensidad",
        "repetición de letras (holaaaaaa)",
        "falta de puntuación y estructura",
        "uso de asteriscos para *énfasis*"
    ]
    
    selected_pattern = random.choice(patterns)
    
    prompt = f"""Genera un diálogo corto (4-6 turnos) donde el usuario muestra este patrón de escritura: {selected_pattern}

El asistente debe:
1. Reconocer el patrón y su significado emocional
2. Adaptar su respuesta al estado emocional inferido
3. Mantener empatía y validación

FORMATO DE SALIDA (JSON):
{{
  "turns": [
    {{"role": "user", "text": "texto con el patrón"}},
    {{"role": "assistant", "text": "respuesta empática"}}
  ],
  "writing_analysis": {{
    "pattern_detected": "{selected_pattern}",
    "emotional_state": "descripción breve",
    "intensity": "baja|media|alta"
  }},
  "risk_class": "NO_CRISIS|LOW_DISTRESS|MODERATE"
}}

GENERA EL DIÁLOGO:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Parsear JSON
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = generated_text[json_start:json_end]
            data = json.loads(json_text)
            
            example = {
                'dialog_id': f'multimodal-writing-{random.randint(10000, 99999)}',
                'turns': data.get('turns', []),
                'labels': {
                    'risk_class': data.get('risk_class', 'NO_CRISIS'),
                    'writing_analysis': data.get('writing_analysis', {}),
                    'source': 'multimodal_generated'
                },
                'meta': {
                    'language': 'es-PE',
                    'domain': ['writing_patterns'],
                    'multimodal_type': 'writing_pattern'
                }
            }
            
            return example
        else:
            return None
    
    except Exception as e:
        print(f"    ✗ Error generando ejemplo de escritura: {e}")
        return None


def generate_longitudinal_example() -> Dict:
    """
    Genera un ejemplo de análisis longitudinal (evolución temporal).
    
    Returns:
        Ejemplo generado
    """
    trajectories = [
        "mejora gradual del estado de ánimo",
        "deterioro progresivo de la motivación",
        "fluctuación entre esperanza y desesperanza",
        "estabilización después de crisis",
        "escalada de ansiedad"
    ]
    
    selected_trajectory = random.choice(trajectories)
    
    prompt = f"""Genera un diálogo más largo (8-12 turnos) que muestre esta evolución temporal: {selected_trajectory}

El diálogo debe mostrar cambios sutiles en el lenguaje y las emociones del usuario a lo largo de la conversación.

El asistente debe:
1. Reconocer la evolución emocional
2. Ajustar su enfoque según la trayectoria
3. Reforzar cambios positivos o intervenir en deterioros

FORMATO DE SALIDA (JSON):
{{
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ],
  "longitudinal_analysis": {{
    "trajectory": "{selected_trajectory}",
    "initial_state": "descripción",
    "final_state": "descripción",
    "key_turning_points": ["punto1", "punto2"]
  }},
  "risk_class": "NO_CRISIS|LOW_DISTRESS|MODERATE"
}}

GENERA EL DIÁLOGO:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        # Parsear JSON
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_text = generated_text[json_start:json_end]
            data = json.loads(json_text)
            
            example = {
                'dialog_id': f'multimodal-longitudinal-{random.randint(10000, 99999)}',
                'turns': data.get('turns', []),
                'labels': {
                    'risk_class': data.get('risk_class', 'NO_CRISIS'),
                    'longitudinal_analysis': data.get('longitudinal_analysis', {}),
                    'source': 'multimodal_generated'
                },
                'meta': {
                    'language': 'es-PE',
                    'domain': ['longitudinal_analysis'],
                    'multimodal_type': 'longitudinal'
                }
            }
            
            return example
        else:
            return None
    
    except Exception as e:
        print(f"    ✗ Error generando ejemplo longitudinal: {e}")
        return None


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("GENERACIÓN DE EJEMPLOS MULTIMODALES")
    print("=" * 70)
    
    # Descargar dataset de emojis
    print(f"\n1. Preparando dataset de emojis...")
    download_emoji_dataset()
    emoji_df = load_emoji_dataset()
    
    # Generar ejemplos
    print(f"\n2. Generando 1,000 ejemplos multimodales...")
    print(f"   Distribución:")
    for example_type, count in EXAMPLE_DISTRIBUTION.items():
        print(f"     → {example_type}: {count} ejemplos")
    print()
    
    all_examples = []
    generation_stats = {t: 0 for t in EXAMPLE_DISTRIBUTION.keys()}
    errors = 0
    
    total_generated = 0
    total_target = sum(EXAMPLE_DISTRIBUTION.values())
    
    # Generar ejemplos de emojis
    print(f"   Generando análisis de emojis...")
    for i in range(EXAMPLE_DISTRIBUTION['emoji_analysis']):
        if (total_generated + 1) % 50 == 0:
            print(f"     Progreso: {total_generated + 1}/{total_target} ejemplos...")
        
        example = generate_emoji_analysis_example(emoji_df)
        
        if example:
            all_examples.append(example)
            generation_stats['emoji_analysis'] += 1
            total_generated += 1
        else:
            errors += 1
        
        time.sleep(0.3)
    
    # Generar ejemplos de patrones de escritura
    print(f"   Generando análisis de patrones de escritura...")
    for i in range(EXAMPLE_DISTRIBUTION['writing_patterns']):
        if (total_generated + 1) % 50 == 0:
            print(f"     Progreso: {total_generated + 1}/{total_target} ejemplos...")
        
        example = generate_writing_pattern_example()
        
        if example:
            all_examples.append(example)
            generation_stats['writing_patterns'] += 1
            total_generated += 1
        else:
            errors += 1
        
        time.sleep(0.3)
    
    # Generar ejemplos longitudinales
    print(f"   Generando análisis longitudinal...")
    for i in range(EXAMPLE_DISTRIBUTION['longitudinal_analysis']):
        if (total_generated + 1) % 50 == 0:
            print(f"     Progreso: {total_generated + 1}/{total_target} ejemplos...")
        
        example = generate_longitudinal_example()
        
        if example:
            all_examples.append(example)
            generation_stats['longitudinal_analysis'] += 1
            total_generated += 1
        else:
            errors += 1
        
        time.sleep(0.3)
    
    # Guardar ejemplos
    print(f"\n3. Guardando ejemplos generados...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Ejemplos guardados en: {OUTPUT_FILE}")
    
    # Guardar metadata
    metadata = {
        'total_generated': len(all_examples),
        'target_total': total_target,
        'generation_stats': generation_stats,
        'errors': errors,
        'model': MODEL,
        'temperature': TEMPERATURE,
        'emoji_dataset': 'Emoji Sentiment Ranking v1.0'
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Ejemplos generados exitosamente: {len(all_examples)}/{total_target}")
    print(f"Errores: {errors}")
    
    print(f"\nDistribución por tipo:")
    for example_type, count in generation_stats.items():
        print(f"  → {example_type}: {count} ejemplos")
    
    print(f"\n✅ Generación completada")
    print(f"   Siguiente paso: Ejecutar Script 8 (smote_implementation.py)")


if __name__ == "__main__":
    main()

