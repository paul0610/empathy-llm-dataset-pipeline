#!/usr/bin/env python3
"""
Script: 5_merge_knowledge_bases.py
Descripción: Fusiona las dos fuentes de datos (Alexander Street y DAIC-WOZ)
             en una base de conocimiento unificada para el sistema RAG.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Carga los chunks clasificados de Alexander Street
2. Carga los diálogos reales de Alexander Street
3. Carga los diálogos traducidos de DAIC-WOZ
4. Fusiona todo en una estructura unificada
5. Guarda la base de conocimiento en formato JSON
"""

import json
import os
from typing import List, Dict
from pathlib import Path

# Configuración
ALEXANDER_CHUNKS_FILE = "/home/ubuntu/alexander_street_chunks_classified.jsonl"
ALEXANDER_DIALOGUES_DIR = "/home/ubuntu/alexander_street_processed/dialogues"
DAICWOZ_DIALOGUES_FILE = "/home/ubuntu/daic_woz_translated.jsonl"
OUTPUT_FILE = "/home/ubuntu/knowledge_base.json"
METADATA_FILE = "/home/ubuntu/knowledge_base_metadata.json"


def load_alexander_chunks() -> List[Dict]:
    """
    Carga los chunks clasificados de Alexander Street.
    
    Returns:
        Lista de chunks
    """
    print("  → Cargando chunks de Alexander Street...")
    
    if not os.path.exists(ALEXANDER_CHUNKS_FILE):
        print(f"    ⚠️  Archivo no encontrado: {ALEXANDER_CHUNKS_FILE}")
        return []
    
    chunks = []
    with open(ALEXANDER_CHUNKS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    
    print(f"    ✓ {len(chunks)} chunks cargados")
    return chunks


def load_alexander_dialogues() -> List[Dict]:
    """
    Carga los diálogos reales de Alexander Street.
    
    Returns:
        Lista de diálogos en formato estructurado
    """
    print("  → Cargando diálogos de Alexander Street...")
    
    if not os.path.exists(ALEXANDER_DIALOGUES_DIR):
        print(f"    ⚠️  Directorio no encontrado: {ALEXANDER_DIALOGUES_DIR}")
        return []
    
    dialogues = []
    dialogue_files = [f for f in os.listdir(ALEXANDER_DIALOGUES_DIR) if f.endswith('.txt')]
    
    for filename in dialogue_files:
        file_path = os.path.join(ALEXANDER_DIALOGUES_DIR, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            dialogue = {
                'dialogue_id': filename.replace('.txt', ''),
                'source': 'alexander_street_dialogue',
                'content': content,
                'language': 'en'
            }
            
            dialogues.append(dialogue)
        
        except Exception as e:
            print(f"    ✗ Error cargando {filename}: {e}")
    
    print(f"    ✓ {len(dialogues)} diálogos cargados")
    return dialogues


def load_daicwoz_dialogues() -> List[Dict]:
    """
    Carga los diálogos traducidos de DAIC-WOZ.
    
    Returns:
        Lista de diálogos
    """
    print("  → Cargando diálogos de DAIC-WOZ...")
    
    if not os.path.exists(DAICWOZ_DIALOGUES_FILE):
        print(f"    ⚠️  Archivo no encontrado: {DAICWOZ_DIALOGUES_FILE}")
        return []
    
    dialogues = []
    with open(DAICWOZ_DIALOGUES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            dialogues.append(json.loads(line))
    
    print(f"    ✓ {len(dialogues)} diálogos cargados")
    return dialogues


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("FUSIÓN DE BASES DE CONOCIMIENTO")
    print("=" * 70)
    
    # Cargar datos
    print("\n1. Cargando datos de las diferentes fuentes...\n")
    
    alexander_chunks = load_alexander_chunks()
    alexander_dialogues = load_alexander_dialogues()
    daicwoz_dialogues = load_daicwoz_dialogues()
    
    # Crear base de conocimiento unificada
    print("\n2. Creando base de conocimiento unificada...")
    
    knowledge_base = {
        'version': '1.0',
        'created_date': '2025-10-09',
        'sources': ['alexander_street', 'daic_woz'],
        'data': {
            'academic_chunks': alexander_chunks,
            'real_dialogues': alexander_dialogues + daicwoz_dialogues
        },
        'statistics': {
            'total_chunks': len(alexander_chunks),
            'total_dialogues': len(alexander_dialogues) + len(daicwoz_dialogues),
            'alexander_dialogues': len(alexander_dialogues),
            'daicwoz_dialogues': len(daicwoz_dialogues)
        }
    }
    
    # Guardar base de conocimiento
    print(f"\n3. Guardando base de conocimiento...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Base de conocimiento guardada en: {OUTPUT_FILE}")
    
    # Calcular estadísticas adicionales
    theme_distribution = {}
    for chunk in alexander_chunks:
        theme = chunk.get('theme', 'unknown')
        theme_distribution[theme] = theme_distribution.get(theme, 0) + 1
    
    # Guardar metadata
    metadata = {
        'total_chunks': len(alexander_chunks),
        'total_dialogues': len(alexander_dialogues) + len(daicwoz_dialogues),
        'sources': {
            'alexander_street': {
                'chunks': len(alexander_chunks),
                'dialogues': len(alexander_dialogues)
            },
            'daic_woz': {
                'dialogues': len(daicwoz_dialogues)
            }
        },
        'theme_distribution': theme_distribution,
        'file_size_mb': os.path.getsize(OUTPUT_FILE) / (1024 * 1024) if os.path.exists(OUTPUT_FILE) else 0
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Base de conocimiento creada exitosamente")
    print(f"\nContenido:")
    print(f"  → Chunks académicos (Alexander Street): {len(alexander_chunks)}")
    print(f"  → Diálogos reales (Alexander Street): {len(alexander_dialogues)}")
    print(f"  → Diálogos traducidos (DAIC-WOZ): {len(daicwoz_dialogues)}")
    print(f"  → Total de diálogos: {len(alexander_dialogues) + len(daicwoz_dialogues)}")
    
    if theme_distribution:
        print(f"\nDistribución temática de chunks:")
        sorted_themes = sorted(theme_distribution.items(), key=lambda x: x[1], reverse=True)
        for theme, count in sorted_themes[:5]:  # Top 5
            print(f"  → {theme}: {count} chunks")
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nTamaño del archivo: {file_size_mb:.2f} MB")
    
    print(f"\n✅ Fusión completada")
    print(f"   Siguiente paso: Ejecutar Script 6 (6_rag_dataset_generator.py)")


if __name__ == "__main__":
    main()

