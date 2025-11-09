#!/usr/bin/env python3
"""
Script 7: Cálculo de Métricas Adicionales
==========================================

Calcula métricas adicionales de generación de texto y clasificación
a partir de los resultados del Script 6 (evaluación con RAG).

Métricas calculadas:
- Generación: Perplexity, Distinct-n, Relevance, Inference Time, Toxicity
- Clasificación: F1, Precision, Recall, Accuracy, Sensitivity, Specificity, ROC-AUC, MCC

Autor: Manus AI
Fecha: 2025-01-16
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import time
from tqdm import tqdm

# Métricas de clasificación
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, matthews_corrcoef,
    classification_report
)

# Sentence embeddings para relevance
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers no disponible. Instalar con: pip3 install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Detección de toxicidad
try:
    from detoxify import Detoxify
    DETOXIFY_AVAILABLE = True
except ImportError:
    print("⚠️  detoxify no disponible. Instalar con: pip3 install detoxify")
    DETOXIFY_AVAILABLE = False

# Transformers para perplexity
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  transformers no disponible. Instalar con: pip3 install transformers")
    TRANSFORMERS_AVAILABLE = False


class MetricsCalculator:
    """Calculador de métricas adicionales"""
    
    def __init__(self, model_path: str = None, device: str = "cpu"):
        """
        Inicializa el calculador de métricas.
        
        Args:
            model_path: Ruta al modelo fine-tuned (para perplexity)
            device: Dispositivo para cálculos (cpu/cuda)
        """
        self.device = device
        self.model = None
        self.tokenizer = None
        self.sentence_model = None
        self.toxicity_model = None
        
        # Cargar modelo para perplexity (opcional)
        if model_path and TRANSFORMERS_AVAILABLE:
            try:
                print(f"📦 Cargando modelo desde {model_path}...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map=device
                )
                self.model.eval()
                print("✅ Modelo cargado correctamente")
            except Exception as e:
                print(f"⚠️  Error cargando modelo: {e}")
                print("   Perplexity no estará disponible")
        
        # Cargar modelo de sentence embeddings
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                print("📦 Cargando modelo de sentence embeddings...")
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✅ Sentence embeddings cargado")
            except Exception as e:
                print(f"⚠️  Error cargando sentence embeddings: {e}")
        
        # Cargar modelo de toxicidad
        if DETOXIFY_AVAILABLE:
            try:
                print("📦 Cargando modelo de detección de toxicidad...")
                self.toxicity_model = Detoxify('multilingual')
                print("✅ Modelo de toxicidad cargado")
            except Exception as e:
                print(f"⚠️  Error cargando modelo de toxicidad: {e}")
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calcula la perplejidad del texto generado.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Perplejidad (menor = mejor)
        """
        if not self.model or not self.tokenizer:
            return None
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            return torch.exp(loss).item()
        except Exception as e:
            print(f"⚠️  Error calculando perplexity: {e}")
            return None
    
    def calculate_distinct_n(self, text: str, n: int = 2) -> float:
        """
        Calcula Distinct-n (diversidad léxica).
        
        Args:
            text: Texto a evaluar
            n: Tamaño del n-grama (1 o 2)
            
        Returns:
            Proporción de n-gramas únicos (mayor = más diverso)
        """
        tokens = text.lower().split()
        if len(tokens) < n:
            return 0.0
        
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        if not ngrams:
            return 0.0
        
        return len(set(ngrams)) / len(ngrams)
    
    def calculate_relevance(self, prompt: str, response: str) -> float:
        """
        Calcula la relevancia semántica entre prompt y respuesta.
        
        Args:
            prompt: Texto del prompt
            response: Respuesta generada
            
        Returns:
            Similitud coseno (0-1, mayor = más relevante)
        """
        if not self.sentence_model:
            return None
        
        try:
            prompt_emb = self.sentence_model.encode(prompt, convert_to_tensor=True)
            response_emb = self.sentence_model.encode(response, convert_to_tensor=True)
            similarity = util.cos_sim(prompt_emb, response_emb).item()
            return similarity
        except Exception as e:
            print(f"⚠️  Error calculando relevance: {e}")
            return None
    
    def calculate_toxicity(self, text: str) -> float:
        """
        Calcula el score de toxicidad del texto.
        
        Args:
            text: Texto a evaluar
            
        Returns:
            Score de toxicidad (0-1, menor = mejor)
        """
        if not self.toxicity_model:
            return None
        
        try:
            results = self.toxicity_model.predict(text)
            return results['toxicity']
        except Exception as e:
            print(f"⚠️  Error calculando toxicity: {e}")
            return None
    
    def calculate_generation_metrics(self, evaluation_results: Dict) -> Dict:
        """
        Calcula todas las métricas de generación de texto.
        
        Args:
            evaluation_results: Resultados del Script 6
            
        Returns:
            Dict con métricas promediadas
        """
        print("\n📊 Calculando métricas de generación...")
        
        metrics = {
            'perplexity': [],
            'distinct_1': [],
            'distinct_2': [],
            'relevance': [],
            'inference_time': [],
            'toxicity': [],
            'response_length_tokens': [],
            'response_length_chars': []
        }
        
        cases = evaluation_results.get('detailed_evaluations', [])
        
        for case in tqdm(cases, desc="Procesando casos"):
            response = case.get('response', '')
            prompt = case.get('prompt', case.get('context', ''))
            
            # Perplexity
            if self.model:
                perp = self.calculate_perplexity(response)
                if perp is not None:
                    metrics['perplexity'].append(perp)
            
            # Distinct-n
            metrics['distinct_1'].append(self.calculate_distinct_n(response, n=1))
            metrics['distinct_2'].append(self.calculate_distinct_n(response, n=2))
            
            # Relevance
            if self.sentence_model:
                rel = self.calculate_relevance(prompt, response)
                if rel is not None:
                    metrics['relevance'].append(rel)
            
            # Inference time (si está disponible en los resultados)
            inf_time = case.get('inference_time', None)
            if inf_time is not None:
                metrics['inference_time'].append(inf_time)
            
            # Toxicity
            if self.toxicity_model:
                tox = self.calculate_toxicity(response)
                if tox is not None:
                    metrics['toxicity'].append(tox)
            
            # Response length
            metrics['response_length_tokens'].append(len(response.split()))
            metrics['response_length_chars'].append(len(response))
        
        # Promediar métricas
        averaged_metrics = {}
        for key, values in metrics.items():
            if values:
                averaged_metrics[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
            else:
                averaged_metrics[key] = None
        
        return averaged_metrics
    
    def calculate_classification_metrics(
        self, 
        y_true: List[int], 
        y_pred: List[int], 
        y_scores: List[float] = None
    ) -> Dict:
        """
        Calcula métricas de clasificación.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Etiquetas predichas
            y_scores: Scores continuos (opcional, para ROC-AUC)
            
        Returns:
            Dict con métricas de clasificación
        """
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['precision'] = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
        metrics['recall'] = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
        metrics['f1_score'] = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Para clasificación binaria
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            # Sensitivity (Recall)
            metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            
            # Specificity
            metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            
            # Positive Predictive Value (Precision)
            metrics['ppv'] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            
            # Negative Predictive Value
            metrics['npv'] = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
            
            # Matthews Correlation Coefficient
            metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
            
            # ROC-AUC (si se proporcionan scores)
            if y_scores is not None:
                try:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
                except:
                    metrics['roc_auc'] = None
        
        return metrics
    
    def calculate_all_classification_metrics(self, evaluation_results: Dict, threshold: float = 3.0) -> Dict:
        """
        Calcula todas las métricas de clasificación.
        
        Args:
            evaluation_results: Resultados del Script 6
            threshold: Umbral para clasificar como "aceptable"
            
        Returns:
            Dict con métricas de clasificación
        """
        print("\n📊 Calculando métricas de clasificación...")
        
        cases = evaluation_results.get('detailed_evaluations', [])
        
        # 1. Clasificación binaria de calidad general
        print("   - Clasificación binaria de calidad...")
        y_pred_quality = [
            1 if case.get('overall_avg', 0) >= threshold else 0 
            for case in cases
        ]
        # Asumimos que todas las respuestas deberían ser aceptables (baseline optimista)
        y_true_quality = [1] * len(y_pred_quality)
        y_scores_quality = [case.get('overall_avg', 0) for case in cases]
        
        quality_metrics = self.calculate_classification_metrics(
            y_true_quality,
            y_pred_quality,
            y_scores_quality
        )
        
        # 2. Detección de crisis (basado en PHQ-8)
        print("   - Detección de crisis...")
        y_pred_crisis = [
            1 if case.get('crisis_detection', 0) >= threshold else 0 
            for case in cases
        ]
        y_true_crisis = [
            1 if case.get('phq8_score', 0) >= 15 else 0  # Severo = crisis
            for case in cases
        ]
        y_scores_crisis = [case.get('crisis_detection', 0) for case in cases]
        
        crisis_metrics = self.calculate_classification_metrics(
            y_true_crisis,
            y_pred_crisis,
            y_scores_crisis
        )
        
        # 3. Clasificación multi-clase por severidad
        print("   - Clasificación multi-clase por severidad...")
        
        def get_severity_class(phq8_score):
            if phq8_score < 5:
                return 0  # Sin depresión
            elif phq8_score < 10:
                return 1  # Leve
            elif phq8_score < 15:
                return 2  # Moderada
            else:
                return 3  # Severa
        
        y_true_severity = [get_severity_class(case.get('phq8_score', 0)) for case in cases]
        
        # Predicción basada en el promedio de síntomas depresivos
        y_pred_severity = []
        for case in cases:
            score = case.get('depression_symptoms', 0)
            if score < 2.5:
                y_pred_severity.append(0)
            elif score < 3.5:
                y_pred_severity.append(1)
            elif score < 4.5:
                y_pred_severity.append(2)
            else:
                y_pred_severity.append(3)
        
        # Métricas multi-clase
        severity_metrics = {
            'accuracy': float(accuracy_score(y_true_severity, y_pred_severity)),
            'macro_f1': float(f1_score(y_true_severity, y_pred_severity, average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y_true_severity, y_pred_severity, average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y_true_severity, y_pred_severity).tolist(),
            'classification_report': classification_report(y_true_severity, y_pred_severity, zero_division=0)
        }
        
        return {
            'binary_quality': quality_metrics,
            'crisis_detection': crisis_metrics,
            'multi_class_severity': severity_metrics
        }


def load_evaluation_results(filepath: str) -> Dict:
    """Carga los resultados del Script 6"""
    print(f"\n📂 Cargando resultados de evaluación desde {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_metrics(metrics: Dict, output_path: str):
    """Guarda las métricas en un archivo JSON"""
    print(f"\n💾 Guardando métricas en {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ Métricas guardadas correctamente")


def print_summary(metrics: Dict):
    """Imprime un resumen de las métricas calculadas"""
    print("\n" + "="*80)
    print("📊 RESUMEN DE MÉTRICAS ADICIONALES")
    print("="*80)
    
    # Métricas de generación
    if 'generation_metrics' in metrics:
        print("\n🔤 MÉTRICAS DE GENERACIÓN:")
        gen = metrics['generation_metrics']
        
        if gen.get('perplexity'):
            print(f"   Perplexity: {gen['perplexity']['mean']:.2f} (±{gen['perplexity']['std']:.2f})")
        if gen.get('distinct_2'):
            print(f"   Distinct-2: {gen['distinct_2']['mean']:.3f} (±{gen['distinct_2']['std']:.3f})")
        if gen.get('relevance'):
            print(f"   Relevance: {gen['relevance']['mean']:.3f} (±{gen['relevance']['std']:.3f})")
        if gen.get('inference_time'):
            print(f"   Inference Time: {gen['inference_time']['mean']:.2f}s (±{gen['inference_time']['std']:.2f}s)")
        if gen.get('toxicity'):
            print(f"   Toxicity: {gen['toxicity']['mean']:.4f} (±{gen['toxicity']['std']:.4f})")
    
    # Métricas de clasificación
    if 'classification_metrics' in metrics:
        print("\n📊 MÉTRICAS DE CLASIFICACIÓN:")
        
        # Calidad binaria
        if 'binary_quality' in metrics['classification_metrics']:
            qual = metrics['classification_metrics']['binary_quality']
            print(f"\n   Calidad Binaria (threshold ≥ 3.0):")
            print(f"      F1-Score: {qual.get('f1_score', 0):.3f}")
            print(f"      Precision: {qual.get('precision', 0):.3f}")
            print(f"      Recall: {qual.get('recall', 0):.3f}")
            print(f"      Accuracy: {qual.get('accuracy', 0):.3f}")
        
        # Detección de crisis
        if 'crisis_detection' in metrics['classification_metrics']:
            crisis = metrics['classification_metrics']['crisis_detection']
            print(f"\n   Detección de Crisis:")
            print(f"      Sensitivity: {crisis.get('sensitivity', 0):.3f}")
            print(f"      Specificity: {crisis.get('specificity', 0):.3f}")
            print(f"      F1-Score: {crisis.get('f1_score', 0):.3f}")
            if crisis.get('roc_auc'):
                print(f"      ROC-AUC: {crisis.get('roc_auc', 0):.3f}")
            print(f"      MCC: {crisis.get('mcc', 0):.3f}")
        
        # Severidad multi-clase
        if 'multi_class_severity' in metrics['classification_metrics']:
            sev = metrics['classification_metrics']['multi_class_severity']
            print(f"\n   Clasificación Multi-Clase (Severidad):")
            print(f"      Macro F1: {sev.get('macro_f1', 0):.3f}")
            print(f"      Weighted F1: {sev.get('weighted_f1', 0):.3f}")
            print(f"      Accuracy: {sev.get('accuracy', 0):.3f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Script 7: Cálculo de Métricas Adicionales"
    )
    parser.add_argument(
        '--evaluation_results',
        type=str,
        required=True,
        help='Ruta al archivo JSON con resultados del Script 6'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Ruta al modelo fine-tuned (opcional, para calcular perplexity)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='additional_metrics.json',
        help='Ruta del archivo de salida (default: additional_metrics.json)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=3.0,
        help='Umbral para clasificación binaria (default: 3.0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Dispositivo para cálculos (default: cpu)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("📊 SCRIPT 7: CÁLCULO DE MÉTRICAS ADICIONALES")
    print("="*80)
    
    # Cargar resultados de evaluación
    evaluation_results = load_evaluation_results(args.evaluation_results)
    
    # Inicializar calculador de métricas
    calculator = MetricsCalculator(model_path=args.model_path, device=args.device)
    
    # Calcular métricas de generación
    generation_metrics = calculator.calculate_generation_metrics(evaluation_results)
    
    # Calcular métricas de clasificación
    classification_metrics = calculator.calculate_all_classification_metrics(
        evaluation_results,
        threshold=args.threshold
    )
    
    # Combinar resultados
    additional_metrics = {
        'model_name': evaluation_results.get('model_name', 'unknown'),
        'evaluation_date': evaluation_results.get('evaluation_date', 'unknown'),
        'threshold': args.threshold,
        'generation_metrics': generation_metrics,
        'classification_metrics': classification_metrics,
        'original_evaluation_file': args.evaluation_results
    }
    
    # Guardar métricas
    save_metrics(additional_metrics, args.output)
    
    # Imprimir resumen
    print_summary(additional_metrics)
    
    print("\n✅ Script 7 completado exitosamente\n")


if __name__ == "__main__":
    main()

