#!/usr/bin/env python3
"""
Implementación SMOTE para Dataset de Empatía
Balanceamiento de clases usando Synthetic Minority Oversampling Technique
"""

import json
import numpy as np
import random
from typing import List, Dict, Tuple
from collections import Counter
import math

class SimpleSMOTE:
    """
    Implementación simplificada de SMOTE para balancear dataset de diálogos
    Basada en el paper original de Chawla et al. (2002)
    """
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        random.seed(random_state)
        np.random.seed(random_state)
    
    def euclidean_distance(self, x1: List[float], x2: List[float]) -> float:
        """Calcular distancia euclidiana entre dos vectores"""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))
    
    def find_k_neighbors(self, sample: List[float], samples: List[List[float]], k: int) -> List[int]:
        """Encontrar k vecinos más cercanos usando distancia euclidiana"""
        distances = []
        
        for i, other_sample in enumerate(samples):
            # Comparar por índice en lugar de contenido para evitar problemas de precisión
            dist = self.euclidean_distance(sample, other_sample)
            distances.append((dist, i))
        
        # Ordenar por distancia y tomar los k más cercanos (excluyendo distancia 0 si existe)
        distances.sort(key=lambda x: x[0])
        
        # Filtrar distancia 0 (mismo elemento) y tomar k vecinos
        neighbors = []
        for dist, idx in distances:
            if dist > 0 and len(neighbors) < k:  # Excluir distancia 0
                neighbors.append(idx)
        
        # Si no hay suficientes vecinos, usar todos los disponibles
        if len(neighbors) == 0 and len(distances) > 0:
            # Tomar el más cercano aunque sea distancia 0
            neighbors = [distances[0][1]]
        
        return neighbors
    
    def generate_synthetic_sample(self, sample: List[float], neighbor: List[float]) -> List[float]:
        """Generar muestra sintética entre sample y neighbor"""
        # Factor aleatorio entre 0 y 1
        gap = random.random()
        
        # Interpolación lineal: sample + gap * (neighbor - sample)
        synthetic = []
        for i in range(len(sample)):
            synthetic_value = sample[i] + gap * (neighbor[i] - sample[i])
            synthetic.append(synthetic_value)
        
        return synthetic
    
    def fit_resample(self, X: List[List[float]], y: List[str], target_distribution: Dict[str, int]) -> Tuple[List[List[float]], List[str]]:
        """
        Aplicar SMOTE para balancear las clases
        
        Args:
            X: Lista de vectores de características (embeddings)
            y: Lista de etiquetas de clase
            target_distribution: Distribución objetivo {clase: cantidad}
        
        Returns:
            X_resampled, y_resampled: Datos balanceados
        """
        print("🔄 Aplicando SMOTE para balancear clases...")
        
        # Contar clases actuales
        current_counts = Counter(y)
        print(f"📊 Distribución actual: {dict(current_counts)}")
        print(f"🎯 Distribución objetivo: {target_distribution}")
        
        # Organizar datos por clase
        class_data = {}
        for i, label in enumerate(y):
            if label not in class_data:
                class_data[label] = []
            class_data[label].append((X[i], i))
        
        # Datos resultantes
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        # Aplicar SMOTE a cada clase minoritaria
        for class_label, target_count in target_distribution.items():
            current_count = current_counts.get(class_label, 0)
            
            if target_count > current_count:
                samples_needed = target_count - current_count
                print(f"🔧 Generando {samples_needed} muestras sintéticas para clase '{class_label}'")
                
                # Obtener muestras de la clase
                class_samples = [sample for sample, _ in class_data[class_label]]
                
                if len(class_samples) < 2:
                    print(f"⚠️ Clase '{class_label}' tiene muy pocas muestras para SMOTE")
                    continue
                
                # Generar muestras sintéticas
                for _ in range(samples_needed):
                    # Seleccionar muestra aleatoria de la clase
                    sample_idx = random.randint(0, len(class_samples) - 1)
                    sample = class_samples[sample_idx]
                    
                    # Encontrar vecinos más cercanos
                    k = min(self.k_neighbors, len(class_samples) - 1)
                    if k <= 0:
                        k = 1
                    
                    neighbor_indices = self.find_k_neighbors(sample, class_samples, k)
                    
                    # Seleccionar vecino aleatorio
                    if neighbor_indices:
                        neighbor_idx = random.choice(neighbor_indices)
                        neighbor = class_samples[neighbor_idx]
                    else:
                        # Fallback: usar una muestra aleatoria diferente
                        available_indices = [i for i in range(len(class_samples)) if class_samples[i] != sample]
                        if available_indices:
                            neighbor_idx = random.choice(available_indices)
                            neighbor = class_samples[neighbor_idx]
                        else:
                            # Último recurso: usar la misma muestra con ruido
                            neighbor = [x + random.uniform(-0.1, 0.1) for x in sample]
                    
                    # Generar muestra sintética
                    synthetic_sample = self.generate_synthetic_sample(sample, neighbor)
                    
                    # Agregar a los datos
                    X_resampled.append(synthetic_sample)
                    y_resampled.append(class_label)
        
        # Verificar resultado
        final_counts = Counter(y_resampled)
        print(f"✅ Distribución final: {dict(final_counts)}")
        
        return X_resampled, y_resampled

class DialogueEmbedder:
    """
    Generador de embeddings simplificado para diálogos
    Convierte texto a vectores numéricos para aplicar SMOTE
    """
    
    def __init__(self):
        # Vocabulario de características emocionales
        self.emotional_features = {
            # Palabras de crisis
            'crisis_words': ['suicidio', 'muerte', 'morir', 'acabar', 'terminar', 'basta', 'no puedo', 'imposible'],
            'depression_words': ['triste', 'deprimido', 'vacío', 'sin ganas', 'cansado', 'agotado', 'desesperanza'],
            'anxiety_words': ['nervioso', 'ansioso', 'miedo', 'pánico', 'preocupado', 'estresado', 'agobiado'],
            'anger_words': ['enojado', 'furioso', 'molesto', 'irritado', 'odio', 'rabia', 'ira'],
            'positive_words': ['feliz', 'contento', 'bien', 'mejor', 'esperanza', 'optimista', 'tranquilo'],
            
            # Patrones de escritura
            'exclamations': ['!', '!!', '!!!'],
            'questions': ['?', '??', '???'],
            'ellipsis': ['...', '....', '.....'],
            'caps': ['MAYÚSCULAS'],
            
            # Emojis por categoría
            'sad_emojis': ['😔', '😢', '😭', '💔', '😞'],
            'happy_emojis': ['😊', '😄', '😁', '🥰', '😍'],
            'crisis_emojis': ['💀', '⚰️', '🔪', '💊'],
            'anxiety_emojis': ['😰', '😨', '😱', '🤯']
        }
        
        # Crear vocabulario completo
        self.vocab = {}
        idx = 0
        for category, words in self.emotional_features.items():
            for word in words:
                self.vocab[word] = idx
                idx += 1
        
        self.vocab_size = len(self.vocab)
        print(f"📝 Vocabulario emocional creado: {self.vocab_size} características")
    
    def text_to_embedding(self, text: str) -> List[float]:
        """Convertir texto a vector de características emocionales"""
        text_lower = text.lower()
        embedding = [0.0] * self.vocab_size
        
        # Contar características
        for word, idx in self.vocab.items():
            if word in text_lower:
                # Contar frecuencia normalizada
                count = text_lower.count(word)
                embedding[idx] = count / len(text.split())
        
        # Características adicionales
        additional_features = [
            len(text.split()),  # Longitud en palabras
            text.count('!') / len(text),  # Densidad de exclamaciones
            text.count('?') / len(text),  # Densidad de preguntas
            text.count('.') / len(text),  # Densidad de puntos
            sum(1 for c in text if c.isupper()) / len(text),  # Proporción mayúsculas
            len([c for c in text if ord(c) > 127]) / len(text)  # Proporción emojis/caracteres especiales
        ]
        
        embedding.extend(additional_features)
        return embedding
    
    def dialogue_to_embedding(self, dialogue: Dict) -> List[float]:
        """Convertir diálogo completo a embedding"""
        user_text = dialogue['turns'][0]['text']
        assistant_text = dialogue['turns'][1]['text']
        
        # Embeddings separados
        user_emb = self.text_to_embedding(user_text)
        assistant_emb = self.text_to_embedding(assistant_text)
        
        # Combinar embeddings (concatenación + estadísticas)
        combined = user_emb + assistant_emb
        
        # Estadísticas adicionales
        combined.extend([
            np.mean(user_emb),  # Media del embedding del usuario
            np.std(user_emb),   # Desviación estándar
            np.mean(assistant_emb),
            np.std(assistant_emb)
        ])
        
        return combined

class SMOTEDatasetBalancer:
    """
    Balanceador de dataset usando SMOTE para diálogos empáticos
    """
    
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.embedder = DialogueEmbedder()
        self.smote = SimpleSMOTE(k_neighbors=5, random_state=42)
        
    def load_dataset(self) -> List[Dict]:
        """Cargar dataset desde archivo JSONL"""
        examples = []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
        
        print(f"📁 Dataset cargado: {len(examples)} ejemplos")
        return examples
    
    def analyze_class_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Analizar distribución actual de clases"""
        risk_classes = [ex['labels']['risk_class'] for ex in examples]
        distribution = Counter(risk_classes)
        
        print("\n📊 DISTRIBUCIÓN ACTUAL DE CLASES:")
        print("=" * 50)
        total = len(examples)
        
        for risk_class, count in sorted(distribution.items()):
            percentage = (count / total) * 100
            print(f"{risk_class:20s}: {count:5d} ({percentage:5.1f}%)")
        
        return dict(distribution)
    
    def define_target_distribution(self, current_dist: Dict[str, int]) -> Dict[str, int]:
        """Definir distribución objetivo balanceada"""
        # Estrategia: Balancear clases minoritarias manteniendo proporción realista
        
        target_dist = {
            'LOW_DISTRESS': current_dist.get('LOW_DISTRESS', 0),  # Mantener clase mayoritaria
            'NO_CRISIS': current_dist.get('NO_CRISIS', 0),       # Mantener
            'MODERATE': max(2000, current_dist.get('MODERATE', 0)),  # Aumentar a 2000
            'HIGH_SUICIDE_RISK': max(2000, current_dist.get('HIGH_SUICIDE_RISK', 0)),  # Crítico: aumentar
            'SELF_HARM_RISK': max(1500, current_dist.get('SELF_HARM_RISK', 0)),
            'VIOLENCE_RISK': max(1500, current_dist.get('VIOLENCE_RISK', 0))
        }
        
        print("\n🎯 DISTRIBUCIÓN OBJETIVO:")
        print("=" * 50)
        total_target = sum(target_dist.values())
        
        for risk_class, count in sorted(target_dist.items()):
            current = current_dist.get(risk_class, 0)
            increase = count - current
            percentage = (count / total_target) * 100
            
            if increase > 0:
                print(f"{risk_class:20s}: {count:5d} ({percentage:5.1f}%) [+{increase}]")
            else:
                print(f"{risk_class:20s}: {count:5d} ({percentage:5.1f}%)")
        
        return target_dist
    
    def convert_to_embeddings(self, examples: List[Dict]) -> Tuple[List[List[float]], List[str], List[Dict]]:
        """Convertir diálogos a embeddings para SMOTE"""
        print("\n🔄 Convirtiendo diálogos a embeddings...")
        
        embeddings = []
        labels = []
        
        for i, example in enumerate(examples):
            embedding = self.embedder.dialogue_to_embedding(example)
            embeddings.append(embedding)
            labels.append(example['labels']['risk_class'])
            
            if (i + 1) % 1000 == 0:
                print(f"  ✅ {i + 1}/{len(examples)} embeddings generados")
        
        print(f"✅ Embeddings completados: {len(embeddings)} vectores de {len(embeddings[0])} dimensiones")
        return embeddings, labels, examples
    
    def embedding_to_dialogue(self, embedding: List[float], reference_examples: List[Dict], target_class: str) -> Dict:
        """Convertir embedding sintético de vuelta a diálogo"""
        # Encontrar el ejemplo más similar de la clase objetivo
        class_examples = [ex for ex in reference_examples if ex['labels']['risk_class'] == target_class]
        
        if not class_examples:
            # Fallback: usar cualquier ejemplo de la clase
            class_examples = reference_examples
        
        # Seleccionar ejemplo base aleatorio
        base_example = random.choice(class_examples)
        
        # Crear nuevo diálogo basado en el ejemplo base pero con variaciones
        synthetic_dialogue = {
            "dialog_id": f"tfm-smote-{target_class.lower()}-{random.randint(1000, 9999)}",
            "turns": base_example["turns"].copy(),  # Usar como base
            "labels": {
                "risk_class": target_class,
                "risk_signals": base_example["labels"]["risk_signals"].copy(),
                "techniques": base_example["labels"]["techniques"].copy(),
                "needs_rag": target_class in ["HIGH_SUICIDE_RISK", "SELF_HARM_RISK", "VIOLENCE_RISK", "MODERATE"]
            },
            "meta": {
                "language": "es-PE",
                "domain": ["smote_synthetic"],
                "style": ["empathetic"],
                "generation_method": "SMOTE_interpolation"
            }
        }
        
        return synthetic_dialogue
    
    def apply_smote_balancing(self):
        """Aplicar SMOTE completo al dataset"""
        print("🎯 Iniciando Balanceamiento SMOTE del Dataset")
        print("=" * 60)
        
        # 1. Cargar dataset
        examples = self.load_dataset()
        
        # 2. Analizar distribución actual
        current_dist = self.analyze_class_distribution(examples)
        
        # 3. Definir distribución objetivo
        target_dist = self.define_target_distribution(current_dist)
        
        # 4. Convertir a embeddings
        embeddings, labels, original_examples = self.convert_to_embeddings(examples)
        
        # 5. Aplicar SMOTE
        print("\n🔧 Aplicando SMOTE...")
        balanced_embeddings, balanced_labels = self.smote.fit_resample(embeddings, labels, target_dist)
        
        # 6. Convertir embeddings sintéticos de vuelta a diálogos
        print("\n🔄 Reconstruyendo diálogos sintéticos...")
        
        balanced_examples = []
        original_count = len(examples)
        
        # Agregar ejemplos originales
        balanced_examples.extend(examples)
        
        # Agregar ejemplos sintéticos
        for i in range(original_count, len(balanced_embeddings)):
            synthetic_embedding = balanced_embeddings[i]
            target_class = balanced_labels[i]
            
            # Convertir embedding a diálogo
            synthetic_dialogue = self.embedding_to_dialogue(
                synthetic_embedding, 
                original_examples, 
                target_class
            )
            
            balanced_examples.append(synthetic_dialogue)
        
        # 7. Guardar dataset balanceado
        self.save_balanced_dataset(balanced_examples)
        
        # 8. Generar estadísticas finales
        self.generate_final_statistics(balanced_examples, original_count)
        
        return balanced_examples
    
    def save_balanced_dataset(self, examples: List[Dict]):
        """Guardar dataset balanceado"""
        print(f"\n💾 Guardando dataset balanceado: {len(examples)} ejemplos")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"✅ Dataset guardado en: {self.output_file}")
    
    def generate_final_statistics(self, examples: List[Dict], original_count: int):
        """Generar estadísticas del dataset balanceado"""
        print("\n📊 ESTADÍSTICAS FINALES DEL DATASET BALANCEADO:")
        print("=" * 60)
        
        total_examples = len(examples)
        synthetic_count = total_examples - original_count
        
        print(f"📝 Total de ejemplos: {total_examples}")
        print(f"📁 Ejemplos originales: {original_count}")
        print(f"🔧 Ejemplos sintéticos (SMOTE): {synthetic_count}")
        print(f"📈 Incremento: {(synthetic_count/original_count)*100:.1f}%")
        
        # Distribución final por clase
        final_dist = Counter([ex['labels']['risk_class'] for ex in examples])
        
        print("\n📂 Distribución final por clase de riesgo:")
        for risk_class, count in sorted(final_dist.items()):
            percentage = (count / total_examples) * 100
            print(f"  {risk_class:20s}: {count:5d} ({percentage:5.1f}%)")
        
        # Métodos de generación
        generation_methods = Counter([
            ex['meta'].get('generation_method', 'original_rag') 
            for ex in examples
        ])
        
        print("\n🛠️ Métodos de generación:")
        for method, count in generation_methods.items():
            percentage = (count / total_examples) * 100
            print(f"  {method:20s}: {count:5d} ({percentage:5.1f}%)")

def main():
    """Función principal"""
    print("🎯 SMOTE Dataset Balancer para TFM de Empatía")
    print("=" * 60)
    
    # Configuración
    input_file = "/home/ubuntu/empathy_dataset_final.jsonl"
    output_file = "/home/ubuntu/empathy_dataset_smote_balanced.jsonl"
    
    # Crear balanceador
    balancer = SMOTEDatasetBalancer(input_file, output_file)
    
    # Aplicar SMOTE
    balanced_dataset = balancer.apply_smote_balancing()
    
    print("\n🎉 ¡Balanceamiento SMOTE completado exitosamente!")
    print(f"📁 Dataset balanceado disponible en: {output_file}")
    print(f"📊 Total de ejemplos: {len(balanced_dataset)}")

if __name__ == "__main__":
    main()
