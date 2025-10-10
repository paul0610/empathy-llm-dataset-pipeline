#!/usr/bin/env python3
"""
Script: 3_classify_chunks_by_theme.py
Descripción: Clasifica cada chunk de Alexander Street por tema clínico
             (ansiedad, depresión, TCC, ACT, MI, crisis, etc.)

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Lee los chunks generados por el Script 2
2. Para cada chunk, usa gpt-4.1-mini para clasificar el tema principal
3. Guarda los chunks con su etiqueta temática
4. Genera estadísticas de distribución temática
"""

import json
from openai import OpenAI
import time
from typing import Dict, List
import os

# Configuración
client = OpenAI()  # API key ya configurada en variables de entorno

INPUT_FILE = "/home/ubuntu/alexander_street_chunks.jsonl"
OUTPUT_FILE = "/home/ubuntu/alexander_street_chunks_classified.jsonl"
METADATA_FILE = "/home/ubuntu/alexander_street_chunks_classification_metadata.json"

# Configuración del modelo
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.0  # Determinista para clasificación consistente

# Categorías temáticas
THEMES = [
    "ansiedad",
    "depresion",
    "tcc",  # Terapia Cognitivo-Conductual
    "act",  # Terapia de Aceptación y Compromiso
    "mi",   # Entrevista Motivacional
    "crisis_e_intervencion",
    "trauma_ptsd",
    "relaciones_pareja",
    "adicciones",
    "otro"
]


def classify_chunk_theme(chunk_text: str, chunk_id: str) -> str:
    """
    Clasifica un chunk en una categoría temática usando gpt-4.1-mini.
    
    Args:
        chunk_text: Texto del chunk
        chunk_id: ID del chunk (para logging)
    
    Returns:
        Tema clasificado
    """
    # Limitar el texto a los primeros 1000 caracteres para ahorrar tokens
    text_sample = chunk_text[:1000]
    
    prompt = f"""Clasifica el siguiente texto clínico/psicológico en UNA de estas categorías:

CATEGORÍAS:
- ansiedad: Trastornos de ansiedad, fobias, ataques de pánico
- depresion: Depresión, trastornos del estado de ánimo
- tcc: Terapia Cognitivo-Conductual (CBT), reestructuración cognitiva
- act: Terapia de Aceptación y Compromiso (ACT), mindfulness
- mi: Entrevista Motivacional, cambio de comportamiento
- crisis_e_intervencion: Crisis suicida, intervención en crisis
- trauma_ptsd: Trauma, PTSD, estrés postraumático
- relaciones_pareja: Terapia de pareja, relaciones interpersonales
- adicciones: Abuso de sustancias, adicciones
- otro: Cualquier otro tema que no encaje en las categorías anteriores

TEXTO:
{text_sample}

INSTRUCCIONES:
- Responde SOLO con el nombre de la categoría (en minúsculas, sin espacios adicionales)
- Si el texto aborda múltiples temas, elige el tema PRINCIPAL
- Si no estás seguro, usa "otro"

CATEGORÍA:"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=20
        )
        
        theme = response.choices[0].message.content.strip().lower()
        
        # Validar que la respuesta es una categoría válida
        if theme not in THEMES:
            print(f"  ⚠️  Categoría inválida '{theme}' para chunk {chunk_id}, usando 'otro'")
            theme = "otro"
        
        return theme
    
    except Exception as e:
        print(f"  ✗ Error clasificando chunk {chunk_id}: {e}")
        return "otro"


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("CLASIFICACIÓN TEMÁTICA DE CHUNKS DE ALEXANDER STREET")
    print("=" * 70)
    
    # Verificar que existe el archivo de entrada
    if not os.path.exists(INPUT_FILE):
        print(f"\n❌ Error: No se encontró el archivo {INPUT_FILE}")
        print("   Ejecuta primero el Script 2 (2_segment_academic_texts.py)")
        return
    
    # Cargar chunks
    print(f"\n1. Cargando chunks desde: {INPUT_FILE}")
    
    chunks = []
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"   Total de chunks cargados: {len(chunks)}")
    
    # Clasificar chunks
    print(f"\n2. Clasificando chunks por tema...")
    print(f"   Modelo: {MODEL}")
    print(f"   Categorías: {', '.join(THEMES)}\n")
    
    classified_chunks = []
    theme_counts = {theme: 0 for theme in THEMES}
    errors = 0
    
    for idx, chunk in enumerate(chunks, 1):
        chunk_id = chunk['chunk_id']
        chunk_text = chunk['text']
        
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{len(chunks)} chunks clasificados...")
        
        # Clasificar
        theme = classify_chunk_theme(chunk_text, chunk_id)
        
        # Añadir tema al chunk
        chunk['theme'] = theme
        classified_chunks.append(chunk)
        theme_counts[theme] += 1
        
        # Pequeña pausa para evitar rate limits
        time.sleep(0.1)
    
    # Guardar chunks clasificados
    print(f"\n3. Guardando chunks clasificados...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in classified_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Chunks clasificados guardados en: {OUTPUT_FILE}")
    
    # Guardar metadata
    metadata = {
        'total_chunks': len(classified_chunks),
        'theme_distribution': theme_counts,
        'theme_percentages': {
            theme: (count / len(classified_chunks) * 100) 
            for theme, count in theme_counts.items()
        },
        'errors': errors
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de chunks clasificados: {len(classified_chunks)}")
    print(f"\nDistribución temática:")
    
    # Ordenar por frecuencia
    sorted_themes = sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)
    
    for theme, count in sorted_themes:
        percentage = (count / len(classified_chunks) * 100) if classified_chunks else 0
        print(f"  → {theme:25s}: {count:5d} chunks ({percentage:5.1f}%)")
    
    print(f"\n✅ Clasificación completada")
    print(f"   Siguiente paso: Ejecutar Script 3.5 (3.5_download_daic_woz_transcripts.py)")


if __name__ == "__main__":
    main()

