#!/usr/bin/env python3
"""
Script: 3.5_download_daic_woz_transcripts.py
Descripción: Descarga y extrae las transcripciones de DAIC-WOZ desde el servidor oficial.
             Solo extrae los archivos XXX_TRANSCRIPT.csv para ahorrar espacio.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Lee los archivos train_split y dev_split para obtener los IDs de participantes
2. Para cada ID, descarga el archivo ZIP correspondiente desde el servidor DAIC-WOZ
3. Extrae únicamente el archivo XXX_TRANSCRIPT.csv del ZIP
4. Guarda las transcripciones en un directorio organizado
"""

import pandas as pd
import requests
import zipfile
import os
from pathlib import Path
from typing import List
import time

# Configuración
BASE_URL = "https://dcapswoz.ict.usc.edu/wwwdaicwoz/"
TRAIN_SPLIT_PATH = "/home/ubuntu/upload/train_split_Depression_AVEC2017.csv"
DEV_SPLIT_PATH = "/home/ubuntu/upload/dev_split_Depression_AVEC2017.csv"
OUTPUT_DIR = "/home/ubuntu/daic_woz_transcripts"
TEMP_DIR = "/home/ubuntu/daic_woz_temp"

# Crear directorios si no existen
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)


def load_participant_ids() -> List[int]:
    """
    Carga los IDs de participantes de train y dev splits.
    
    Returns:
        Lista de IDs de participantes a procesar
    """
    # Leer train split
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    train_ids = train_df['Participant_ID'].tolist()
    print(f"Train split: {len(train_ids)} participantes")
    
    # Leer dev split
    dev_df = pd.read_csv(DEV_SPLIT_PATH)
    dev_ids = dev_df['Participant_ID'].tolist()
    print(f"Dev split: {len(dev_ids)} participantes")
    
    # Combinar y ordenar
    all_ids = sorted(train_ids + dev_ids)
    print(f"Total: {len(all_ids)} participantes")
    
    return all_ids


def download_zip(participant_id: int, output_path: str) -> bool:
    """
    Descarga el archivo ZIP de un participante desde el servidor DAIC-WOZ.
    
    Args:
        participant_id: ID del participante
        output_path: Ruta donde guardar el ZIP
    
    Returns:
        True si la descarga fue exitosa, False en caso contrario
    """
    zip_filename = f"{participant_id}_P.zip"
    url = BASE_URL + zip_filename
    
    try:
        print(f"  → Descargando {zip_filename}...", end=" ")
        response = requests.get(url, stream=True, timeout=300)
        
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            total_mb = total_size / (1024 * 1024)
            
            with open(output_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            print(f"✓ ({total_mb:.1f} MB)")
            return True
        else:
            print(f"✗ (HTTP {response.status_code})")
            return False
    
    except Exception as e:
        print(f"✗ (Error: {e})")
        return False


def extract_transcript(zip_path: str, participant_id: int, output_dir: str) -> bool:
    """
    Extrae únicamente el archivo TRANSCRIPT.csv del ZIP.
    
    Args:
        zip_path: Ruta al archivo ZIP
        participant_id: ID del participante
        output_dir: Directorio donde guardar la transcripción
    
    Returns:
        True si la extracción fue exitosa, False en caso contrario
    """
    transcript_filename = f"{participant_id}_TRANSCRIPT.csv"
    # El archivo dentro del ZIP está en una carpeta con el nombre del participante
    transcript_in_zip = f"{participant_id}_P/{transcript_filename}"
    
    try:
        print(f"  → Extrayendo {transcript_filename}...", end=" ")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Verificar que el archivo existe en el ZIP
            if transcript_in_zip in zip_ref.namelist():
                # Extraer el archivo
                zip_ref.extract(transcript_in_zip, TEMP_DIR)
                
                # Mover al directorio de salida con nombre simplificado
                source_path = os.path.join(TEMP_DIR, transcript_in_zip)
                dest_path = os.path.join(output_dir, transcript_filename)
                
                os.rename(source_path, dest_path)
                
                # Limpiar directorio temporal
                temp_participant_dir = os.path.join(TEMP_DIR, f"{participant_id}_P")
                if os.path.exists(temp_participant_dir):
                    os.rmdir(temp_participant_dir)
                
                print("✓")
                return True
            else:
                print(f"✗ (No encontrado en ZIP)")
                return False
    
    except Exception as e:
        print(f"✗ (Error: {e})")
        return False


def process_participant(participant_id: int) -> bool:
    """
    Procesa un participante: descarga el ZIP y extrae la transcripción.
    
    Args:
        participant_id: ID del participante
    
    Returns:
        True si el procesamiento fue exitoso, False en caso contrario
    """
    print(f"\nParticipant {participant_id}:")
    
    # Verificar si ya existe la transcripción
    transcript_path = os.path.join(OUTPUT_DIR, f"{participant_id}_TRANSCRIPT.csv")
    if os.path.exists(transcript_path):
        print(f"  ✓ Transcripción ya existe, saltando...")
        return True
    
    # Descargar ZIP
    zip_path = os.path.join(TEMP_DIR, f"{participant_id}_P.zip")
    
    if not download_zip(participant_id, zip_path):
        return False
    
    # Extraer transcripción
    success = extract_transcript(zip_path, participant_id, OUTPUT_DIR)
    
    # Eliminar ZIP para ahorrar espacio
    try:
        os.remove(zip_path)
        print(f"  → ZIP eliminado para ahorrar espacio")
    except:
        pass
    
    return success


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("DESCARGA Y EXTRACCIÓN DE TRANSCRIPCIONES DAIC-WOZ")
    print("=" * 70)
    
    # Cargar IDs de participantes
    print("\n1. Cargando IDs de participantes...")
    participant_ids = load_participant_ids()
    
    # Procesar cada participante
    print(f"\n2. Descargando y extrayendo transcripciones...")
    print(f"   Directorio de salida: {OUTPUT_DIR}")
    print(f"   Directorio temporal: {TEMP_DIR}\n")
    
    successful = []
    failed = []
    
    for idx, participant_id in enumerate(participant_ids, 1):
        print(f"[{idx}/{len(participant_ids)}]", end=" ")
        
        if process_participant(participant_id):
            successful.append(participant_id)
        else:
            failed.append(participant_id)
        
        # Pequeña pausa para no sobrecargar el servidor
        time.sleep(0.5)
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Transcripciones descargadas exitosamente: {len(successful)}")
    print(f"Transcripciones con errores: {len(failed)}")
    
    if failed:
        print(f"\nParticipantes con errores: {failed}")
        print("\nPuedes volver a ejecutar el script para reintentar las descargas fallidas.")
    
    print(f"\n✅ Transcripciones guardadas en: {OUTPUT_DIR}")
    print(f"   Total de archivos: {len(successful)}")
    
    # Limpiar directorio temporal
    try:
        if os.path.exists(TEMP_DIR) and not os.listdir(TEMP_DIR):
            os.rmdir(TEMP_DIR)
            print(f"   Directorio temporal eliminado")
    except:
        pass


if __name__ == "__main__":
    main()

