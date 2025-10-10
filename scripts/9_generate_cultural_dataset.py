#!/usr/bin/env python3
"""
Script 9: Generación de Dataset de Adaptación Cultural con Modismos Peruanos

Este script genera 2,000 diálogos sintéticos incorporando modismos peruanos
de forma natural, manteniendo la distribución estratificada de categorías
del dataset principal.

Autor: [Tu Nombre]
Fecha: 30 de octubre de 2025
Repositorio: https://github.com/paul0610/empathy-llm-dataset-pipeline
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Rutas de archivos
KNOWLEDGE_BASE_PATH = "data/processed/knowledge_base_merged.jsonl"
MODISMOS_PATH = "data/modismos_peruanos.json"
OUTPUT_PATH = "data/processed/cultural_dataset.jsonl"

# Parámetros de generación
TOTAL_DIALOGUES = 2000
TEMPERATURE = 0.7
MAX_TOKENS = 1500
MODEL = "gpt-4.1-mini"  # o "gpt-4o-mini" según disponibilidad

# Distribución de categorías (mantener proporciones del dataset principal)
CATEGORY_DISTRIBUTION = {
    "empathy_training": 0.467,      # 46.7% -> 934 diálogos
    "therapeutic_techniques": 0.267, # 26.7% -> 534 diálogos
    "safety_crisis": 0.133,          # 13.3% -> 266 diálogos
    "confounding_cases": 0.100,      # 10.0% -> 200 diálogos
    "general_conversation": 0.033    # 3.3%  -> 66 diálogos
}

# Mapeo de categorías a categorías emocionales
CATEGORY_TO_EMOTIONS = {
    "empathy_training": ["Tristeza", "Ansiedad", "Neutral", "Confusión"],
    "therapeutic_techniques": ["Tristeza", "Ansiedad", "Estrés", "Neutral"],
    "safety_crisis": ["Tristeza", "Desesperanza", "Enojo", "Confusión"],
    "confounding_cases": ["Neutral", "Alegría", "Confusión"],
    "general_conversation": ["Neutral", "Alegría", "Tristeza"]
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def load_knowledge_base(path: str) -> List[Dict[str, Any]]:
    """
    Carga la base de conocimiento unificada (Alexander Street + DAIC-WOZ).
    
    Args:
        path: Ruta al archivo JSONL de la base de conocimiento
        
    Returns:
        Lista de diccionarios con los chunks de conocimiento
    """
    knowledge_base = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            knowledge_base.append(json.loads(line))
    print(f"✓ Base de conocimiento cargada: {len(knowledge_base)} chunks")
    return knowledge_base


def load_modismos(path: str) -> List[Dict[str, str]]:
    """
    Carga la tabla de modismos peruanos.
    
    Args:
        path: Ruta al archivo JSON con los modismos
        
    Returns:
        Lista de diccionarios con modismos
    """
    with open(path, 'r', encoding='utf-8') as f:
        modismos = json.load(f)
    print(f"✓ Modismos cargados: {len(modismos)} expresiones")
    return modismos


def filter_modismos_by_emotion(
    modismos: List[Dict[str, str]], 
    emotions: List[str], 
    n: int = 7
) -> List[Dict[str, str]]:
    """
    Filtra modismos por categoría emocional.
    
    Args:
        modismos: Lista completa de modismos
        emotions: Lista de categorías emocionales permitidas
        n: Número máximo de modismos a retornar
        
    Returns:
        Lista filtrada de modismos
    """
    filtered = [m for m in modismos if m["categoria_emocional"] in emotions]
    return random.sample(filtered, min(n, len(filtered)))


def build_tfidf_index(knowledge_base: List[Dict[str, Any]]) -> tuple:
    """
    Construye índice TF-IDF para recuperación de contexto.
    
    Args:
        knowledge_base: Lista de chunks de conocimiento
        
    Returns:
        Tupla (vectorizer, tfidf_matrix)
    """
    texts = [chunk["text"] for chunk in knowledge_base]
    
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=None  # Mantener stop words para español
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"✓ Índice TF-IDF construido: {tfidf_matrix.shape}")
    
    return vectorizer, tfidf_matrix


def retrieve_context(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix: np.ndarray,
    knowledge_base: List[Dict[str, Any]],
    top_k: int = 3
) -> List[str]:
    """
    Recupera los top-k chunks más relevantes para una consulta.
    
    Args:
        query: Consulta de búsqueda
        vectorizer: Vectorizador TF-IDF entrenado
        tfidf_matrix: Matriz TF-IDF de la base de conocimiento
        knowledge_base: Lista de chunks de conocimiento
        top_k: Número de chunks a recuperar
        
    Returns:
        Lista de textos de los chunks más relevantes
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    return [knowledge_base[i]["text"] for i in top_indices]


def build_generation_prompt(
    category: str,
    scenario: str,
    context: List[str],
    modismos: List[Dict[str, str]]
) -> str:
    """
    Construye el prompt de generación con modismos filtrados.
    
    Args:
        category: Categoría del diálogo
        scenario: Descripción del escenario
        context: Lista de chunks de contexto recuperados
        modismos: Lista de modismos filtrados
        
    Returns:
        Prompt completo para GPT-4.1-mini
    """
    # Formatear modismos
    modismos_text = "\n".join([
        f"  - \"{m['modismo']}\" ({m['español_estandar']}) - {m['contexto']}"
        for m in modismos
    ])
    
    # Formatear contexto
    context_text = "\n\n".join([f"[Fragmento {i+1}]\n{c}" for i, c in enumerate(context)])
    
    prompt = f"""Eres un experto en psicología clínica y salud mental. Tu tarea es generar un diálogo realista entre un joven peruano (18-25 años, residente de la región costa) y un asistente virtual de apoyo emocional llamado Aura.

CATEGORÍA: {category}
ESCENARIO: {scenario}

CONOCIMIENTO RELEVANTE:
{context_text}

ADAPTACIÓN CULTURAL:
El paciente es un joven peruano de Lima. Usa modismos peruanos de forma natural (1-2 por diálogo):

{modismos_text}

INSTRUCCIONES:
1. Genera un diálogo de 6-10 turnos (3-5 intercambios)
2. El usuario debe expresar una preocupación emocional realista usando modismos peruanos naturales
3. El asistente debe responder con empatía, validación y técnicas apropiadas
4. Usa lenguaje natural y coloquial (español peruano)
5. Incluye emociones sutiles y matices
6. Los modismos deben integrarse de forma natural, sin forzar su uso

FORMATO DE SALIDA (JSON):
{{
  "turns": [
    {{"role": "user", "text": "..."}},
    {{"role": "assistant", "text": "..."}}
  ],
  "risk_class": "NO_CRISIS|LOW_DISTRESS|MODERATE|HIGH_SUICIDE_RISK",
  "risk_signals": ["señal1", "señal2"],
  "domain": ["dominio1"]
}}

GENERA EL DIÁLOGO:"""
    
    return prompt


def generate_dialogue(
    client: OpenAI,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 1500
) -> Dict[str, Any]:
    """
    Genera un diálogo usando GPT-4.1-mini.
    
    Args:
        client: Cliente de OpenAI
        prompt: Prompt de generación
        temperature: Temperatura de generación
        max_tokens: Máximo de tokens a generar
        
    Returns:
        Diccionario con el diálogo generado
    """
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Eres un experto en psicología clínica y generación de datos sintéticos para entrenamiento de modelos de IA en salud mental."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        dialogue = json.loads(content)
        
        return dialogue
    
    except Exception as e:
        print(f"✗ Error al generar diálogo: {e}")
        return None


def calculate_category_counts(total: int, distribution: Dict[str, float]) -> Dict[str, int]:
    """
    Calcula el número de diálogos por categoría según la distribución.
    
    Args:
        total: Número total de diálogos a generar
        distribution: Diccionario con proporciones por categoría
        
    Returns:
        Diccionario con conteos por categoría
    """
    counts = {}
    remaining = total
    
    # Calcular conteos
    for category, proportion in distribution.items():
        count = int(total * proportion)
        counts[category] = count
        remaining -= count
    
    # Distribuir el residuo en la categoría más grande
    if remaining > 0:
        max_category = max(counts, key=counts.get)
        counts[max_category] += remaining
    
    return counts


# ============================================================================
# ESCENARIOS POR CATEGORÍA
# ============================================================================

SCENARIOS = {
    "empathy_training": [
        "El usuario se siente solo y necesita alguien que lo escuche",
        "El usuario está pasando por una ruptura amorosa",
        "El usuario se siente incomprendido por su familia",
        "El usuario está experimentando tristeza sin motivo aparente",
        "El usuario se siente abrumado por las expectativas sociales"
    ],
    "therapeutic_techniques": [
        "El usuario tiene pensamientos negativos recurrentes sobre sí mismo",
        "El usuario experimenta ansiedad antes de eventos sociales",
        "El usuario tiene dificultades para concentrarse en sus estudios",
        "El usuario evita situaciones que le generan ansiedad",
        "El usuario tiene patrones de pensamiento catastrófico"
    ],
    "safety_crisis": [
        "El usuario expresa ideación suicida pasiva",
        "El usuario menciona autolesiones recientes",
        "El usuario se siente sin esperanza sobre el futuro",
        "El usuario ha tenido pensamientos de hacerse daño",
        "El usuario expresa desesperanza y falta de sentido de vida"
    ],
    "confounding_cases": [
        "El usuario está estresado pero funcionando bien",
        "El usuario tiene un mal día pero no presenta síntomas clínicos",
        "El usuario está preocupado por un examen pero no tiene ansiedad clínica",
        "El usuario está triste por una situación puntual pero no deprimido",
        "El usuario busca consejo sobre una decisión importante"
    ],
    "general_conversation": [
        "El usuario quiere conversar sobre su día",
        "El usuario busca motivación para sus metas",
        "El usuario quiere hablar sobre sus hobbies",
        "El usuario busca consejos para mejorar su bienestar",
        "El usuario quiere compartir un logro reciente"
    ]
}


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """
    Función principal del script.
    """
    print("=" * 80)
    print("GENERACIÓN DE DATASET DE ADAPTACIÓN CULTURAL CON MODISMOS PERUANOS")
    print("=" * 80)
    print()
    
    # Verificar que existen los archivos necesarios
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print(f"✗ Error: No se encontró la base de conocimiento en {KNOWLEDGE_BASE_PATH}")
        print("  Ejecuta primero el script 5_merge_knowledge_bases.py")
        return
    
    if not os.path.exists(MODISMOS_PATH):
        print(f"✗ Error: No se encontró la tabla de modismos en {MODISMOS_PATH}")
        print("  Crea el archivo data/modismos_peruanos.json con la tabla de modismos")
        return
    
    # Cargar datos
    print("Cargando datos...")
    knowledge_base = load_knowledge_base(KNOWLEDGE_BASE_PATH)
    modismos = load_modismos(MODISMOS_PATH)
    print()
    
    # Construir índice TF-IDF
    print("Construyendo índice TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf_index(knowledge_base)
    print()
    
    # Inicializar cliente de OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Calcular conteos por categoría
    category_counts = calculate_category_counts(TOTAL_DIALOGUES, CATEGORY_DISTRIBUTION)
    
    print("Distribución de diálogos por categoría:")
    for category, count in category_counts.items():
        percentage = (count / TOTAL_DIALOGUES) * 100
        print(f"  - {category}: {count} ({percentage:.1f}%)")
    print()
    
    # Crear directorio de salida si no existe
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    # Generar diálogos
    print(f"Generando {TOTAL_DIALOGUES} diálogos...")
    print()
    
    generated_dialogues = []
    
    for category, count in category_counts.items():
        print(f"Generando {count} diálogos de categoría '{category}'...")
        
        for i in tqdm(range(count), desc=category):
            # Seleccionar escenario aleatorio
            scenario = random.choice(SCENARIOS[category])
            
            # Filtrar modismos por categoría emocional
            emotions = CATEGORY_TO_EMOTIONS[category]
            filtered_modismos = filter_modismos_by_emotion(modismos, emotions, n=7)
            
            # Recuperar contexto relevante
            query = f"{category} {scenario}"
            context = retrieve_context(query, vectorizer, tfidf_matrix, knowledge_base, top_k=3)
            
            # Construir prompt
            prompt = build_generation_prompt(category, scenario, context, filtered_modismos)
            
            # Generar diálogo
            dialogue = generate_dialogue(client, prompt, TEMPERATURE, MAX_TOKENS)
            
            if dialogue:
                # Añadir metadatos
                dialogue["category"] = category
                dialogue["scenario"] = scenario
                dialogue["modismos_used"] = [m["modismo"] for m in filtered_modismos]
                
                generated_dialogues.append(dialogue)
            else:
                print(f"  ✗ Error al generar diálogo {i+1}/{count}")
        
        print()
    
    # Guardar diálogos
    print(f"Guardando {len(generated_dialogues)} diálogos en {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for dialogue in generated_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
    
    print()
    print("=" * 80)
    print(f"✓ Generación completada: {len(generated_dialogues)} diálogos guardados")
    print("=" * 80)
    print()
    print("Estadísticas finales:")
    print(f"  - Total de diálogos generados: {len(generated_dialogues)}")
    print(f"  - Archivo de salida: {OUTPUT_PATH}")
    print(f"  - Modismos utilizados: {len(modismos)}")
    print()
    print("Próximos pasos:")
    print("  1. Validar manualmente el 10% del dataset (200 diálogos)")
    print("  2. Fusionar con el dataset principal (15,000 diálogos)")
    print("  3. Aplicar SMOTE al dataset fusionado")
    print()


if __name__ == "__main__":
    main()

