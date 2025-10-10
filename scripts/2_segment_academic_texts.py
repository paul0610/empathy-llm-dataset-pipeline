#!/usr/bin/env python3
"""
Script: 2_segment_academic_texts.py
Descripción: Segmenta los textos académicos de Alexander Street en chunks de ~500 palabras
             para su uso en el sistema RAG.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Lee los textos académicos clasificados por el Script 1
2. Utiliza LangChain RecursiveCharacterTextSplitter para segmentar
3. Crea chunks de ~500 palabras con overlap de 50 palabras
4. Guarda los chunks en formato JSON para procesamiento posterior
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuración
ACADEMIC_DIR = "/home/ubuntu/alexander_street_processed/academic"
OUTPUT_FILE = "/home/ubuntu/alexander_street_chunks.jsonl"
METADATA_FILE = "/home/ubuntu/alexander_street_chunks_metadata.json"

# Parámetros de segmentación
CHUNK_SIZE_WORDS = 500
OVERLAP_WORDS = 50
AVG_CHARS_PER_WORD = 5  # Promedio de caracteres por palabra en inglés

# Calcular tamaños en caracteres
CHUNK_SIZE = CHUNK_SIZE_WORDS * AVG_CHARS_PER_WORD  # ~2,500 caracteres
OVERLAP_SIZE = OVERLAP_WORDS * AVG_CHARS_PER_WORD    # ~250 caracteres


def clean_text(text: str) -> str:
    """
    Limpia el texto eliminando timestamps y marcadores de transcripción.
    
    Args:
        text: Texto original
    
    Returns:
        Texto limpio
    """
    import re
    
    # Eliminar timestamps (formato 00:00:00)
    text = re.sub(r'\d{2}:\d{2}:\d{2}', '', text)
    
    # Eliminar marcadores de transcripción
    text = re.sub(r'TRANSCRIPT OF VIDEO FILE:', '', text)
    text = re.sub(r'BEGIN TRANSCRIPT:', '', text)
    text = re.sub(r'END TRANSCRIPT:', '', text)
    text = re.sub(r'_+', '', text)  # Líneas de guiones bajos
    
    # Eliminar líneas vacías múltiples
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r' +', ' ', text)
    
    return text.strip()


def segment_document(file_path: str, splitter: RecursiveCharacterTextSplitter) -> List[Dict]:
    """
    Segmenta un documento en chunks.
    
    Args:
        file_path: Ruta al archivo
        splitter: Instancia de RecursiveCharacterTextSplitter
    
    Returns:
        Lista de diccionarios con los chunks y metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Limpiar texto
        content = clean_text(content)
        
        # Segmentar
        chunks = splitter.split_text(content)
        
        # Crear objetos de chunk con metadata
        filename = os.path.basename(file_path)
        chunk_objects = []
        
        for idx, chunk_text in enumerate(chunks):
            # Contar palabras
            word_count = len(chunk_text.split())
            
            chunk_obj = {
                'chunk_id': f"{filename.replace('.txt', '')}_{idx:03d}",
                'source_file': filename,
                'chunk_index': idx,
                'text': chunk_text,
                'word_count': word_count,
                'char_count': len(chunk_text),
                'source': 'alexander_street'
            }
            
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    except Exception as e:
        print(f"  ✗ Error segmentando {os.path.basename(file_path)}: {e}")
        return []


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("SEGMENTACIÓN DE TEXTOS ACADÉMICOS DE ALEXANDER STREET")
    print("=" * 70)
    
    # Verificar que existe el directorio de entrada
    if not os.path.exists(ACADEMIC_DIR):
        print(f"\n❌ Error: No se encontró el directorio {ACADEMIC_DIR}")
        print("   Ejecuta primero el Script 1 (1_classify_alexander_street.py)")
        return
    
    # Listar archivos
    print(f"\n1. Buscando archivos académicos en: {ACADEMIC_DIR}")
    
    academic_files = [f for f in os.listdir(ACADEMIC_DIR) if f.endswith('.txt')]
    print(f"   Total de archivos encontrados: {len(academic_files)}")
    
    # Configurar splitter
    print(f"\n2. Configurando segmentador...")
    print(f"   Tamaño de chunk: ~{CHUNK_SIZE_WORDS} palabras ({CHUNK_SIZE} caracteres)")
    print(f"   Overlap: ~{OVERLAP_WORDS} palabras ({OVERLAP_SIZE} caracteres)")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    # Segmentar archivos
    print(f"\n3. Segmentando archivos...\n")
    
    all_chunks = []
    file_stats = []
    
    for idx, filename in enumerate(academic_files, 1):
        file_path = os.path.join(ACADEMIC_DIR, filename)
        
        if idx % 100 == 0:
            print(f"   Progreso: {idx}/{len(academic_files)} archivos procesados...")
        
        chunks = segment_document(file_path, splitter)
        
        if chunks:
            all_chunks.extend(chunks)
            file_stats.append({
                'filename': filename,
                'num_chunks': len(chunks),
                'total_words': sum(c['word_count'] for c in chunks)
            })
    
    # Guardar chunks
    print(f"\n4. Guardando chunks...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Chunks guardados en: {OUTPUT_FILE}")
    
    # Guardar metadata
    metadata = {
        'total_files': len(academic_files),
        'total_chunks': len(all_chunks),
        'avg_chunks_per_file': len(all_chunks) / len(academic_files) if academic_files else 0,
        'chunk_size_words': CHUNK_SIZE_WORDS,
        'overlap_words': OVERLAP_WORDS,
        'file_stats': file_stats,
        'word_count_distribution': {
            'min': min(c['word_count'] for c in all_chunks) if all_chunks else 0,
            'max': max(c['word_count'] for c in all_chunks) if all_chunks else 0,
            'avg': sum(c['word_count'] for c in all_chunks) / len(all_chunks) if all_chunks else 0
        }
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Archivos académicos procesados: {len(academic_files)}")
    print(f"Total de chunks generados: {len(all_chunks)}")
    print(f"Promedio de chunks por archivo: {len(all_chunks) / len(academic_files):.1f}")
    
    if all_chunks:
        print(f"\nDistribución de palabras por chunk:")
        print(f"  → Mínimo: {metadata['word_count_distribution']['min']} palabras")
        print(f"  → Máximo: {metadata['word_count_distribution']['max']} palabras")
        print(f"  → Promedio: {metadata['word_count_distribution']['avg']:.1f} palabras")
    
    print(f"\n✅ Segmentación completada")
    print(f"   Siguiente paso: Ejecutar Script 3 (3_classify_chunks_by_theme.py)")


if __name__ == "__main__":
    main()

