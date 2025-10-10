#!/usr/bin/env python3
"""
Script: 1_classify_alexander_street.py
Descripción: Clasifica los 1,330 documentos de Alexander Street en dos categorías:
             - Diálogos reales (transcripciones de sesiones terapéuticas)
             - Textos académicos (conferencias, presentaciones, artículos)

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Lee todos los archivos .txt del directorio de Alexander Street
2. Analiza el contenido para detectar patrones de diálogo
3. Clasifica cada archivo según la presencia de turnos conversacionales
4. Guarda los resultados en directorios separados
"""

import os
import re
from pathlib import Path
from typing import Tuple, Dict
import shutil
import json

# Configuración
INPUT_DIR = "/home/ubuntu/alexander_street_data/AlexanderStreet/AlexanderStreet"
OUTPUT_DIR = "/home/ubuntu/alexander_street_processed"
DIALOGUES_DIR = os.path.join(OUTPUT_DIR, "dialogues")
ACADEMIC_DIR = os.path.join(OUTPUT_DIR, "academic")
METADATA_FILE = os.path.join(OUTPUT_DIR, "classification_metadata.json")

# Crear directorios de salida
Path(DIALOGUES_DIR).mkdir(parents=True, exist_ok=True)
Path(ACADEMIC_DIR).mkdir(parents=True, exist_ok=True)


def detect_dialogue_patterns(content: str) -> Tuple[bool, int, Dict]:
    """
    Detecta si un archivo contiene diálogos terapéuticos basándose en patrones.
    
    Args:
        content: Contenido del archivo
    
    Returns:
        Tupla con (es_dialogo, num_turnos, estadisticas)
    """
    # Patrones de diálogo terapéutico
    dialogue_patterns = [
        r'THERAPIST:',
        r'CLIENT:',
        r'PATIENT:',
        r'COUNSELOR:',
        r'DR\.\s+\w+:',  # Dr. NOMBRE:
        r'INTERVIEWER:',
        r'PARTICIPANT:',
        r'\b[A-Z]{2,}\s+[A-Z]+:',  # NOMBRE APELLIDO:
    ]
    
    # Contar ocurrencias de cada patrón
    pattern_counts = {}
    total_turns = 0
    
    for pattern in dialogue_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        count = len(matches)
        if count > 0:
            pattern_counts[pattern] = count
            total_turns += count
    
    # Detectar líneas que parecen turnos conversacionales
    # Formato típico: "SPEAKER: texto"
    turn_pattern = r'^[A-Z\s\.]+:\s+.+$'
    turn_lines = re.findall(turn_pattern, content, re.MULTILINE)
    turn_count = len(turn_lines)
    
    # Estadísticas adicionales
    stats = {
        'pattern_counts': pattern_counts,
        'turn_lines': turn_count,
        'total_turns': total_turns,
        'content_length': len(content),
        'has_timestamps': bool(re.search(r'\d{2}:\d{2}:\d{2}', content))
    }
    
    # Criterios de clasificación:
    # - Más de 10 turnos conversacionales identificados
    # - O más de 5 líneas con formato de turno
    is_dialogue = (total_turns > 10) or (turn_count > 5)
    
    return is_dialogue, total_turns, stats


def classify_file(file_path: str) -> Tuple[str, Dict]:
    """
    Clasifica un archivo individual.
    
    Args:
        file_path: Ruta al archivo
    
    Returns:
        Tupla con (categoria, metadata)
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        is_dialogue, num_turns, stats = detect_dialogue_patterns(content)
        
        category = 'dialogue' if is_dialogue else 'academic'
        
        metadata = {
            'filename': os.path.basename(file_path),
            'category': category,
            'num_turns': num_turns,
            'stats': stats
        }
        
        return category, metadata
    
    except Exception as e:
        print(f"  ✗ Error procesando {os.path.basename(file_path)}: {e}")
        return 'error', {'filename': os.path.basename(file_path), 'error': str(e)}


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("CLASIFICACIÓN DE ARCHIVOS DE ALEXANDER STREET")
    print("=" * 70)
    
    # Listar archivos
    print(f"\n1. Buscando archivos en: {INPUT_DIR}")
    
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.txt')]
    print(f"   Total de archivos encontrados: {len(all_files)}")
    
    # Clasificar archivos
    print(f"\n2. Clasificando archivos...\n")
    
    dialogues = []
    academic = []
    errors = []
    metadata_list = []
    
    for idx, filename in enumerate(all_files, 1):
        file_path = os.path.join(INPUT_DIR, filename)
        
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{len(all_files)} archivos procesados...")
        
        category, metadata = classify_file(file_path)
        metadata_list.append(metadata)
        
        if category == 'dialogue':
            dialogues.append(filename)
            # Copiar a directorio de diálogos
            shutil.copy2(file_path, os.path.join(DIALOGUES_DIR, filename))
        elif category == 'academic':
            academic.append(filename)
            # Copiar a directorio académico
            shutil.copy2(file_path, os.path.join(ACADEMIC_DIR, filename))
        else:
            errors.append(filename)
    
    # Guardar metadata
    print(f"\n3. Guardando metadata...")
    
    metadata_summary = {
        'total_files': len(all_files),
        'dialogues': len(dialogues),
        'academic': len(academic),
        'errors': len(errors),
        'dialogue_files': dialogues,
        'academic_files': academic,
        'error_files': errors,
        'detailed_metadata': metadata_list
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata_summary, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total de archivos procesados: {len(all_files)}")
    print(f"  → Diálogos reales: {len(dialogues)} ({len(dialogues)/len(all_files)*100:.1f}%)")
    print(f"  → Textos académicos: {len(academic)} ({len(academic)/len(all_files)*100:.1f}%)")
    print(f"  → Errores: {len(errors)}")
    
    print(f"\nArchivos guardados en:")
    print(f"  → Diálogos: {DIALOGUES_DIR}")
    print(f"  → Académicos: {ACADEMIC_DIR}")
    
    # Mostrar algunos ejemplos de diálogos
    if dialogues:
        print(f"\nEjemplos de diálogos identificados:")
        for filename in dialogues[:5]:
            print(f"  - {filename}")
    
    print("\n✅ Clasificación completada")


if __name__ == "__main__":
    main()

