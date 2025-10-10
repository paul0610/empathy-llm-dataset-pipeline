#!/usr/bin/env python3
"""
Script: 6_rag_dataset_generator.py
Descripción: Genera 15,000 diálogos sintéticos usando RAG (Retrieval-Augmented Generation).
             Utiliza TF-IDF para recuperación y gpt-4.1-mini para generación.

Autor: Generado para TFM - VIU
Fecha: 9 de octubre de 2025

Metodología:
1. Carga la base de conocimiento unificada
2. Indexa los chunks usando TF-IDF (scikit-learn)
3. Para cada diálogo a generar:
   a. Recupera 2-3 chunks relevantes según el tema
   b. Selecciona 1-2 diálogos de ejemplo (few-shot)
   c. Genera un nuevo diálogo usando gpt-4.1-mini
4. Guarda los 15,000 diálogos generados en formato JSONL
"""

import json
import random
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from typing import List, Dict, Tuple
import os

# Configuración
client = OpenAI()  # API key ya configurada en variables de entorno

KNOWLEDGE_BASE_FILE = "/home/ubuntu/knowledge_base.json"
OUTPUT_FILE = "/home/ubuntu/rag_synthetic_dialogues.jsonl"
METADATA_FILE = "/home/ubuntu/rag_generation_metadata.json"

# Configuración del modelo
MODEL = "gpt-4.1-mini"
TEMPERATURE = 0.8  # Creatividad moderada

# Distribución de ejemplos a generar
GENERATION_PLAN = {
    'empathy_training': 7000,
    'cbt_techniques': 4000,
    'crisis_safety': 2000,
    'confounders': 1500,
    'general': 500
}

TOTAL_EXAMPLES = sum(GENERATION_PLAN.values())  # 15,000


class RAGGenerator:
    """
    Generador de diálogos sintéticos usando RAG.
    """
    
    def __init__(self, knowledge_base: Dict):
        """
        Inicializa el generador RAG.
        
        Args:
            knowledge_base: Base de conocimiento cargada
        """
        self.chunks = knowledge_base['data']['academic_chunks']
        self.dialogues = knowledge_base['data']['real_dialogues']
        
        print(f"  → Inicializando RAG con {len(self.chunks)} chunks y {len(self.dialogues)} diálogos")
        
        # Indexar chunks con TF-IDF
        print(f"  → Indexando chunks con TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        chunk_texts = [c['text'] for c in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
        
        print(f"  → Indexación completada")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Recupera los chunks más relevantes para una query.
        
        Args:
            query: Query de búsqueda
            top_k: Número de chunks a recuperar
        
        Returns:
            Lista de chunks relevantes
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_indices]
    
    def select_example_dialogues(self, num_examples: int = 2) -> List[Dict]:
        """
        Selecciona diálogos de ejemplo aleatoriamente.
        
        Args:
            num_examples: Número de ejemplos a seleccionar
        
        Returns:
            Lista de diálogos de ejemplo
        """
        return random.sample(self.dialogues, min(num_examples, len(self.dialogues)))
    
    def generate_dialogue(self, category: str, scenario_description: str) -> Dict:
        """
        Genera un diálogo sintético usando RAG.
        
        Args:
            category: Categoría del diálogo (empathy_training, cbt_techniques, etc.)
            scenario_description: Descripción del escenario a generar
        
        Returns:
            Diálogo generado en formato JSON
        """
        # Recuperar chunks relevantes
        relevant_chunks = self.retrieve_relevant_chunks(scenario_description, top_k=2)
        
        # Seleccionar ejemplos
        example_dialogues = self.select_example_dialogues(num_examples=1)
        
        # Construir contexto
        context_text = "\n\n".join([
            f"CONOCIMIENTO {i+1}:\n{chunk['text'][:500]}"
            for i, chunk in enumerate(relevant_chunks)
        ])
        
        # Construir prompt
        prompt = self._build_generation_prompt(
            category, scenario_description, context_text, example_dialogues
        )
        
        # Generar diálogo
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            # Parsear el diálogo generado
            dialogue = self._parse_generated_dialogue(generated_text, category)
            
            return dialogue
        
        except Exception as e:
            print(f"    ✗ Error generando diálogo: {e}")
            return None
    
    def _build_generation_prompt(self, category: str, scenario: str, 
                                 context: str, examples: List[Dict]) -> str:
        """
        Construye el prompt para generación.
        """
        prompt = f"""Eres un experto en psicología clínica y salud mental. Tu tarea es generar un diálogo realista entre un usuario (que busca apoyo emocional) y un asistente de IA empático.

CATEGORÍA: {category}
ESCENARIO: {scenario}

CONOCIMIENTO RELEVANTE:
{context}

INSTRUCCIONES:
1. Genera un diálogo de 6-10 turnos (3-5 intercambios)
2. El usuario debe expresar una preocupación emocional realista
3. El asistente debe responder con empatía, validación y técnicas apropiadas
4. Usa lenguaje natural y coloquial (español latinoamericano)
5. Incluye emociones sutiles y matices

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
    
    def _parse_generated_dialogue(self, generated_text: str, category: str) -> Dict:
        """
        Parsea el texto generado en formato JSON.
        """
        try:
            # Intentar extraer JSON del texto
            json_start = generated_text.find('{')
            json_end = generated_text.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = generated_text[json_start:json_end]
                dialogue_data = json.loads(json_text)
                
                # Añadir metadata
                dialogue_id = f"rag-{category}-{random.randint(10000, 99999)}"
                
                dialogue = {
                    'dialog_id': dialogue_id,
                    'turns': dialogue_data.get('turns', []),
                    'labels': {
                        'risk_class': dialogue_data.get('risk_class', 'NO_CRISIS'),
                        'risk_signals': dialogue_data.get('risk_signals', []),
                        'category': category,
                        'source': 'rag_generated'
                    },
                    'meta': {
                        'language': 'es-PE',
                        'domain': dialogue_data.get('domain', ['general']),
                        'generation_method': 'rag_tfidf_gpt4.1mini'
                    }
                }
                
                return dialogue
            else:
                return None
        
        except Exception as e:
            print(f"    ✗ Error parseando diálogo: {e}")
            return None


def generate_scenario_description(category: str) -> str:
    """
    Genera una descripción de escenario según la categoría.
    
    Args:
        category: Categoría del diálogo
    
    Returns:
        Descripción del escenario
    """
    scenarios = {
        'empathy_training': [
            "Usuario expresa ansiedad por exámenes universitarios",
            "Usuario se siente solo y aislado socialmente",
            "Usuario tiene problemas para dormir por estrés laboral",
            "Usuario se siente abrumado por responsabilidades familiares",
            "Usuario experimenta tristeza después de una ruptura"
        ],
        'cbt_techniques': [
            "Usuario tiene pensamientos negativos automáticos sobre sí mismo",
            "Usuario catastrofiza sobre el futuro",
            "Usuario necesita reestructuración cognitiva para ansiedad",
            "Usuario practica técnicas de exposición gradual",
            "Usuario identifica distorsiones cognitivas"
        ],
        'crisis_safety': [
            "Usuario expresa ideación suicida pasiva",
            "Usuario tiene pensamientos de autolesión",
            "Usuario en crisis emocional aguda",
            "Usuario necesita plan de seguridad",
            "Usuario expresa desesperanza severa"
        ],
        'confounders': [
            "Usuario hace preguntas fuera del ámbito de salud mental",
            "Usuario solicita consejos médicos específicos",
            "Usuario intenta manipular al asistente",
            "Usuario expresa contenido inapropiado",
            "Usuario busca validación de comportamientos de riesgo"
        ],
        'general': [
            "Usuario busca técnicas de relajación",
            "Usuario pregunta sobre higiene del sueño",
            "Usuario necesita estrategias de manejo del estrés",
            "Usuario busca información sobre mindfulness",
            "Usuario pregunta sobre hábitos saludables"
        ]
    }
    
    return random.choice(scenarios.get(category, scenarios['general']))


def main():
    """
    Función principal del script.
    """
    print("=" * 70)
    print("GENERACIÓN DE DIÁLOGOS SINTÉTICOS CON RAG")
    print("=" * 70)
    
    # Verificar archivo de entrada
    if not os.path.exists(KNOWLEDGE_BASE_FILE):
        print(f"\n❌ Error: No se encontró {KNOWLEDGE_BASE_FILE}")
        print("   Ejecuta primero el Script 5 (5_merge_knowledge_bases.py)")
        return
    
    # Cargar base de conocimiento
    print(f"\n1. Cargando base de conocimiento...")
    
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    print(f"   ✓ Base de conocimiento cargada")
    
    # Inicializar generador RAG
    print(f"\n2. Inicializando generador RAG...")
    
    generator = RAGGenerator(knowledge_base)
    
    # Generar diálogos
    print(f"\n3. Generando {TOTAL_EXAMPLES} diálogos sintéticos...")
    print(f"   Distribución:")
    for category, count in GENERATION_PLAN.items():
        print(f"     → {category}: {count} ejemplos")
    print()
    
    generated_dialogues = []
    generation_stats = {cat: 0 for cat in GENERATION_PLAN.keys()}
    errors = 0
    
    total_generated = 0
    
    for category, target_count in GENERATION_PLAN.items():
        print(f"   Generando categoría: {category}")
        
        for i in range(target_count):
            if (total_generated + 1) % 100 == 0:
                print(f"     Progreso total: {total_generated + 1}/{TOTAL_EXAMPLES} diálogos...")
            
            scenario = generate_scenario_description(category)
            dialogue = generator.generate_dialogue(category, scenario)
            
            if dialogue:
                generated_dialogues.append(dialogue)
                generation_stats[category] += 1
                total_generated += 1
            else:
                errors += 1
            
            # Pausa para evitar rate limits
            time.sleep(0.5)
    
    # Guardar diálogos
    print(f"\n4. Guardando diálogos generados...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for dialogue in generated_dialogues:
            f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
    
    print(f"   ✓ Diálogos guardados en: {OUTPUT_FILE}")
    
    # Guardar metadata
    metadata = {
        'total_generated': len(generated_dialogues),
        'target_total': TOTAL_EXAMPLES,
        'generation_stats': generation_stats,
        'errors': errors,
        'model': MODEL,
        'temperature': TEMPERATURE,
        'retrieval_method': 'tfidf',
        'top_k_chunks': 2
    }
    
    with open(METADATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"   ✓ Metadata guardada en: {METADATA_FILE}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    print(f"Diálogos generados exitosamente: {len(generated_dialogues)}/{TOTAL_EXAMPLES}")
    print(f"Errores: {errors}")
    
    print(f"\nDistribución por categoría:")
    for category, count in generation_stats.items():
        print(f"  → {category}: {count} diálogos")
    
    print(f"\n✅ Generación completada")
    print(f"   Siguiente paso: Ejecutar Script 7 (7_generate_multimodal_examples.py)")


if __name__ == "__main__":
    main()

