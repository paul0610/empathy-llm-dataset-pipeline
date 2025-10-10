#!/usr/bin/env python3
"""
Script: 4_process_daic_woz.py
Descripción: Procesa las entrevistas de DAIC-WOZ, las traduce al español peruano
             y las convierte al formato JSON del proyecto.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025
"""

import pandas as pd
import json
import os
from pathlib import Path
from openai import OpenAI
import time
from typing import List, Dict, Optional

# Configuración
client = OpenAI()  # API key ya configurada en variables de entorno

# Rutas
TRAIN_SPLIT_PATH = "/home/ubuntu/upload/train_split_Depression_AVEC2017.csv"
DEV_SPLIT_PATH = "/home/ubuntu/upload/dev_split_Depression_AVEC2017.csv"
TRANSCRIPTS_DIR = "/home/ubuntu/daic_woz_transcripts/"  # Directorio con transcripciones extraídas
OUTPUT_FILE = "/home/ubuntu/daic_woz_translated.jsonl"

# Configuración de traducción
TRANSLATION_MODEL = "gpt-4.1-mini"
TRANSLATION_TEMPERATURE = 0.3  # Baja temperatura para consistencia


def map_phq8_to_risk_class(phq8_score: int) -> str:
    """
    Mapea un score PHQ-8 a nuestra clasificación de riesgo.
    
    Args:
        phq8_score: Puntuación PHQ-8 (0-24)
    
    Returns:
        Clase de riesgo: NO_CRISIS, LOW_DISTRESS, MODERATE, HIGH_SUICIDE_RISK
    """
    if phq8_score <= 4:
        return "NO_CRISIS"
    elif phq8_score <= 9:
        return "LOW_DISTRESS"
    elif phq8_score <= 14:
        return "MODERATE"
    else:  # 15-24
        return "HIGH_SUICIDE_RISK"


def infer_risk_signals(phq8_score: int) -> List[str]:
    """
    Infiere señales de riesgo basadas en el score PHQ-8.
    
    Args:
        phq8_score: Puntuación PHQ-8 (0-24)
    
    Returns:
        Lista de señales de riesgo
    """
    signals = []
    
    if phq8_score >= 10:
        signals.append("depresion_clinica")
    
    if phq8_score >= 15:
        signals.append("depresion_severa")
        signals.append("riesgo_suicida")
    elif phq8_score >= 10:
        signals.append("depresion_moderada")
    elif phq8_score >= 5:
        signals.append("depresion_leve")
    
    return signals


def read_transcript(file_path: str) -> pd.DataFrame:
    """
    Lee un archivo TRANSCRIPT.csv de DAIC-WOZ.
    
    Args:
        file_path: Ruta al archivo de transcripción
    
    Returns:
        DataFrame con las columnas: start_time, stop_time, speaker, value
    """
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            names=['start_time', 'stop_time', 'speaker', 'value'],
            encoding='utf-8',
            engine='python'
        )
        return df
    except Exception as e:
        print(f"Error leyendo {file_path}: {e}")
        return None


def translate_dialogue(dialogue_text: str, phq8_score: int, participant_id: int) -> str:
    """
    Traduce un diálogo completo del inglés al español peruano usando gpt-4.1-mini.
    
    Args:
        dialogue_text: Texto del diálogo en inglés
        phq8_score: Score PHQ-8 del participante (para contexto)
        participant_id: ID del participante
    
    Returns:
        Diálogo traducido en español peruano
    """
    prompt = f"""Traduce la siguiente entrevista clínica del inglés al español latinoamericano (variante peruana).

CONTEXTO:
- Esta es una entrevista de detección de depresión del dataset DAIC-WOZ
- Participant ID: {participant_id}
- PHQ-8 Score: {phq8_score} (0-24, donde >= 10 indica depresión clínica)

INSTRUCCIONES IMPORTANTES:
1. Preserva el tono emocional de cada intervención
2. Usa expresiones naturales del español peruano (evita modismos de otros países)
3. Mantén la estructura de turnos conversacionales exactamente como está
4. NO traduzcas el nombre "Ellie" (es el nombre de la entrevistadora virtual)
5. Mantén las pausas y vacilaciones (um, uh, etc.) como "eh", "este", "mmm"
6. Preserva la naturalidad y espontaneidad del habla
7. NO añadas ni quites información, solo traduce

FORMATO DE SALIDA:
Devuelve SOLO el diálogo traducido, manteniendo el formato:
[Speaker]: [texto traducido]

DIÁLOGO ORIGINAL:
{dialogue_text}

DIÁLOGO TRADUCIDO:"""

    try:
        response = client.chat.completions.create(
            model=TRANSLATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TRANSLATION_TEMPERATURE
        )
        
        translated_text = response.choices[0].message.content.strip()
        return translated_text
    
    except Exception as e:
        print(f"Error en traducción para Participant {participant_id}: {e}")
        return None


def format_dialogue_for_translation(df: pd.DataFrame) -> str:
    """
    Formatea un DataFrame de transcripción en texto para traducción.
    
    Args:
        df: DataFrame con las columnas start_time, stop_time, speaker, value
    
    Returns:
        Texto formateado para traducción
    """
    lines = []
    for _, row in df.iterrows():
        speaker = row['speaker']
        text = row['value']
        lines.append(f"{speaker}: {text}")
    
    return "\n".join(lines)


def parse_translated_dialogue(translated_text: str) -> List[Dict[str, str]]:
    """
    Parsea el diálogo traducido y lo convierte en una lista de turnos.
    
    Args:
        translated_text: Texto del diálogo traducido
    
    Returns:
        Lista de diccionarios con 'role' y 'text'
    """
    turns = []
    lines = translated_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Buscar el formato "Speaker: text"
        if ':' in line:
            parts = line.split(':', 1)
            speaker = parts[0].strip()
            text = parts[1].strip()
            
            # Mapear speaker a nuestro formato
            if speaker.lower() == 'ellie':
                role = 'interviewer'
            elif speaker.lower() in ['participant', 'participante']:
                role = 'user'
            else:
                # Si no reconocemos el speaker, intentar inferir
                role = 'interviewer' if 'ellie' in speaker.lower() else 'user'
            
            turns.append({
                'role': role,
                'text': text
            })
    
    return turns


def process_participant(participant_id: int, phq8_score: int, phq8_binary: int, 
                       gender: int, transcript_path: str) -> Optional[Dict]:
    """
    Procesa un participante completo: lee transcripción, traduce y formatea.
    
    Args:
        participant_id: ID del participante
        phq8_score: Score PHQ-8
        phq8_binary: Etiqueta binaria PHQ-8 (0 o 1)
        gender: Género (0=femenino, 1=masculino)
        transcript_path: Ruta al archivo de transcripción
    
    Returns:
        Diccionario con el diálogo procesado en nuestro formato
    """
    print(f"\nProcesando Participant {participant_id} (PHQ-8: {phq8_score})...")
    
    # Leer transcripción
    df = read_transcript(transcript_path)
    if df is None or df.empty:
        print(f"  ❌ No se pudo leer la transcripción")
        return None
    
    print(f"  ✓ Transcripción leída: {len(df)} turnos")
    
    # Formatear para traducción
    dialogue_text = format_dialogue_for_translation(df)
    
    # Traducir
    print(f"  → Traduciendo con {TRANSLATION_MODEL}...")
    translated_text = translate_dialogue(dialogue_text, phq8_score, participant_id)
    
    if translated_text is None:
        print(f"  ❌ Error en traducción")
        return None
    
    print(f"  ✓ Traducción completada")
    
    # Parsear diálogo traducido
    turns = parse_translated_dialogue(translated_text)
    
    if not turns:
        print(f"  ❌ No se pudieron parsear los turnos traducidos")
        return None
    
    print(f"  ✓ {len(turns)} turnos parseados")
    
    # Mapear a nuestro formato
    risk_class = map_phq8_to_risk_class(phq8_score)
    risk_signals = infer_risk_signals(phq8_score)
    
    dialogue = {
        'dialog_id': f'daic-woz-{participant_id}',
        'turns': turns,
        'labels': {
            'risk_class': risk_class,
            'phq8_score': phq8_score,
            'risk_signals': risk_signals,
            'source': 'daic_woz_translated'
        },
        'meta': {
            'language': 'es-PE',
            'domain': ['clinical_interview'],
            'original_language': 'en',
            'gender': gender,
            'phq8_binary': phq8_binary
        }
    }
    
    return dialogue


def load_participant_data() -> pd.DataFrame:
    """
    Carga y combina los datos de train y dev splits.
    
    Returns:
        DataFrame con todos los participantes (train + dev)
    """
    # Leer train split
    train_df = pd.read_csv(TRAIN_SPLIT_PATH)
    print(f"Train split: {len(train_df)} participantes")
    
    # Leer dev split
    dev_df = pd.read_csv(DEV_SPLIT_PATH)
    print(f"Dev split: {len(dev_df)} participantes")
    
    # Combinar
    combined_df = pd.concat([train_df, dev_df], ignore_index=True)
    print(f"Total: {len(combined_df)} participantes")
    
    return combined_df


def main(test_mode: bool = True, test_participant_id: int = 303):
    """
    Función principal del script.
    
    Args:
        test_mode: Si True, solo procesa un participante de prueba
        test_participant_id: ID del participante para modo de prueba
    """
    print("=" * 70)
    print("PROCESAMIENTO DE DAIC-WOZ")
    print("=" * 70)
    
    # Cargar datos de participantes
    print("\n1. Cargando datos de participantes...")
    participants_df = load_participant_data()
    
    if test_mode:
        print(f"\n⚠️  MODO DE PRUEBA: Solo procesando Participant {test_participant_id}")
        participants_df = participants_df[participants_df['Participant_ID'] == test_participant_id]
        
        if participants_df.empty:
            print(f"❌ Participant {test_participant_id} no encontrado en los datos")
            return
    
    # Procesar participantes
    print(f"\n2. Procesando {len(participants_df)} participante(s)...\n")
    
    processed_dialogues = []
    errors = []
    
    for idx, row in participants_df.iterrows():
        participant_id = row['Participant_ID']
        phq8_score = row['PHQ8_Score']
        phq8_binary = row['PHQ8_Binary']
        gender = row['Gender']
        
        # Construir ruta al archivo de transcripción
        transcript_filename = f"{participant_id}_TRANSCRIPT.csv"
        transcript_path = os.path.join(TRANSCRIPTS_DIR, transcript_filename)
        
        # Verificar que existe el archivo
        if not os.path.exists(transcript_path):
            print(f"⚠️  Archivo no encontrado: {transcript_filename}")
            errors.append(participant_id)
            continue
        
        # Procesar participante
        dialogue = process_participant(
            participant_id, phq8_score, phq8_binary, gender, transcript_path
        )
        
        if dialogue:
            processed_dialogues.append(dialogue)
            print(f"  ✅ Participant {participant_id} procesado exitosamente")
        else:
            errors.append(participant_id)
            print(f"  ❌ Error procesando Participant {participant_id}")
        
        # Pequeña pausa para evitar rate limits
        time.sleep(1)
    
    # Guardar resultados
    print(f"\n3. Guardando resultados...")
    
    output_path = OUTPUT_FILE if not test_mode else OUTPUT_FILE.replace('.jsonl', '_test.jsonl')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for dialogue in processed_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
    
    print(f"  ✓ Guardado en: {output_path}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Total procesados exitosamente: {len(processed_dialogues)}")
    print(f"Total con errores: {len(errors)}")
    
    if errors:
        print(f"\nParticipantes con errores: {errors}")
    
    print("\n✅ Proceso completado")


if __name__ == "__main__":
    # MODO DE PRUEBA: Primero probar con Participant 300
    print("\n🧪 Ejecutando en MODO DE PRUEBA con Participant 303...")
    print("Si funciona correctamente, cambiar test_mode=False para procesar todos.\n")
    
    main(test_mode=True, test_participant_id=303)
    
    # Para procesar todos los participantes, descomentar la siguiente línea:
    # main(test_mode=False)

