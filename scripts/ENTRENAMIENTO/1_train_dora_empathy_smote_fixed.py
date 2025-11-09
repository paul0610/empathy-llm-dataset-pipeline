"""
Fine-tuning Llama 3.2 1B Instruct con DoRA - Dataset de Empatía Balanceado con SMOTE
Optimizado para RTX 3080 Ti Laptop GPU con soporte CUDA
VERSIÓN FINAL: Sin Flash Attention + Solución para conflicto PEFT/Gradient Checkpointing
"""

import torch
import json
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import os
import time
from datetime import datetime

# Configuración CUDA para RTX 3080 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True  # Optimización para hardware consistente

def load_empathy_dataset(jsonl_file):
    """Cargar y procesar el dataset de empatía balanceado con SMOTE"""
    
    print("📂 Cargando dataset de empatía balanceado con SMOTE...")
    print(f"📁 Archivo: {jsonl_file}")
    
    data = []
    
    # Cargar JSONL línea por línea
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                data.append(example)
                
                # Progreso cada 1000 líneas
                if line_num % 1000 == 0:
                    print(f"  ✅ Cargadas {line_num} conversaciones...")
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ Error en línea {line_num}: {e}")
                continue
    
    print(f"✅ Dataset cargado: {len(data)} conversaciones")
    
    # Estadísticas del dataset
    risk_classes = {}
    generation_methods = {}
    
    for example in data:
        risk_class = example['labels']['risk_class']
        risk_classes[risk_class] = risk_classes.get(risk_class, 0) + 1
        
        generation_method = example['meta'].get('generation_method', 'original_rag')
        generation_methods[generation_method] = generation_methods.get(generation_method, 0) + 1
    
    print("\n📊 Distribución por clase de riesgo:")
    for risk_class, count in sorted(risk_classes.items()):
        percentage = (count / len(data)) * 100
        print(f"  {risk_class:20s}: {count:5d} ({percentage:5.1f}%)")
    
    print("\n🛠️ Métodos de generación:")
    for method, count in generation_methods.items():
        percentage = (count / len(data)) * 100
        print(f"  {method:20s}: {count:5d} ({percentage:5.1f}%)")
    
    # Convertir a formato Dataset
    dataset = Dataset.from_list(data)
    
    return dataset

def format_empathy_chat_template(example, tokenizer):
    """Convertir diálogos de empatía a formato de chat de Llama"""
    
    messages = []
    
    # Extraer turnos del diálogo
    for turn in example["turns"]:
        if turn["role"] == "user":
            messages.append({"role": "user", "content": turn["text"]})
        elif turn["role"] == "assistant":
            messages.append({"role": "assistant", "content": turn["text"]})
    
    # Aplicar template de chat de Llama
    try:
        formatted = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        return {"text": formatted}
    except Exception as e:
        print(f"⚠️ Error formateando diálogo {example.get('dialog_id', 'unknown')}: {e}")
        # Fallback: formato simple
        user_text = example["turns"][0]["text"]
        assistant_text = example["turns"][1]["text"]
        fallback_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_text}<|eot_id|>"
        return {"text": fallback_text}

def setup_dora_model_optimized():
    """Configurar modelo con DoRA optimizado para RTX 3080 Ti (SOLUCIÓN PEFT/Gradient Checkpointing)"""
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    
    print(f"🚀 Cargando modelo: {model_name}")
    print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Cargar modelo base SIN Flash Attention
    print("⚡ Cargando modelo base (sin Flash Attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16 para eficiencia en RTX 3080 Ti
        device_map={"": 0},         # Forzar GPU 0
        trust_remote_code=True,
        # Sin attn_implementation para evitar Flash Attention
    )
    
    # Cargar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # SOLUCIÓN: Preparar modelo para PEFT antes de aplicar DoRA
    print("🔧 Preparando modelo para PEFT...")
    model = prepare_model_for_kbit_training(model)
    
    # Configuración DoRA optimizada para empatía y detección de crisis
    dora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,                    # Rank aumentado para mejor capacidad de empatía
        lora_alpha=64,           # Alpha (2x rank) para estabilidad
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",     # MLP layers
        ],
        lora_dropout=0.05,       # Dropout bajo para preservar conocimiento empático
        use_dora=True,           # ¡Activar DoRA!
        bias="none",
    )
    
    # Aplicar DoRA
    print("⚡ Aplicando configuración DoRA...")
    model = get_peft_model(model, dora_config)
    
    # SOLUCIÓN: Habilitar gradient checkpointing DESPUÉS de aplicar PEFT
    print("💾 Habilitando gradient checkpointing compatible con PEFT...")
    model.enable_input_require_grads()  # Crucial para PEFT + gradient checkpointing
    
    # Mostrar parámetros entrenables
    print("\n📊 Parámetros del modelo:")
    model.print_trainable_parameters()
    
    return model, tokenizer

def create_training_args_optimized():
    """Crear argumentos de entrenamiento optimizados para RTX 3080 Ti"""
    
    # Timestamp para identificar el entrenamiento
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./empathy-llama-dora-smote_{timestamp}"
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Configuración de épocas y batch
        num_train_epochs=3,                    # 3 épocas para dataset grande
        per_device_train_batch_size=8,         # Volver a 8 con la solución PEFT
        gradient_accumulation_steps=2,         # Volver a configuración original
        
        # Configuración de aprendizaje
        learning_rate=5e-5,                    # LR ligeramente más alto para DoRA
        lr_scheduler_type="cosine",            # Cosine annealing
        warmup_steps=100,                     # Warmup corto (3% del entrenamiento)
        weight_decay=0.01,                     # Regularización ligera
        
        # Optimizaciones de memoria y velocidad
        fp16=True,                             # FP16 para RTX 3080 Ti
        dataloader_drop_last=True,             # Consistencia en batch size
        dataloader_num_workers=4,              # Volver a configuración original
        
        # Logging y guardado
        logging_steps=25,                      # Log cada 25 steps
        save_steps=500,                        # Guardar cada 500 steps
        save_total_limit=3,                    # Mantener solo 3 checkpoints
        eval_strategy="no",              # Sin evaluación para velocidad
        
        # Configuraciones adicionales
        report_to=None,                        # Sin wandb/tensorboard
        remove_unused_columns=False,           # Preservar metadatos
        label_smoothing_factor=0.1,            # Suavizado para mejor generalización
        
        # SOLUCIÓN: Gradient checkpointing compatible con PEFT
        gradient_checkpointing=True,           # Ahora funciona con la preparación PEFT
        
        # Optimizaciones específicas para CUDA
        bf16=False,                            # Usar FP16 en lugar de BF16
    )

def main():
    """Función principal de entrenamiento"""
    
    print("🎯 Entrenamiento DoRA - Dataset de Empatía con SMOTE (VERSIÓN FINAL)")
    print("=" * 70)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA no disponible. Este script requiere GPU NVIDIA.")
        return
    
    print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
    print(f"🔧 Solución aplicada: PEFT + Gradient Checkpointing compatible")
    print(f"⚠️ Usando implementación estándar de atención (sin Flash Attention)")
    
    start_time = time.time()
    
    try:
        # 1. Configurar modelo DoRA
        print("\n🔧 Configurando modelo DoRA...")
        model, tokenizer = setup_dora_model_optimized()
        
        # 2. Cargar dataset
        print("\n📂 Cargando dataset...")
        dataset = load_empathy_dataset("empathy_dataset_smote_balanced.jsonl")
        
        # 3. Procesar dataset
        print("\n🔧 Procesando dataset...")
        def tokenize_function(examples):
            # Aplicar template de chat
            formatted_texts = []
            
            for i in range(len(examples["turns"])):
                example = {
                    "turns": examples["turns"][i],
                    "dialog_id": examples["dialog_id"][i]
                }
                formatted = format_empathy_chat_template(example, tokenizer)
                formatted_texts.append(formatted["text"])
            
            # Tokenizar con longitud máxima optimizada
            tokenized = tokenizer(
                formatted_texts,
                truncation=True,
                padding=False,
                max_length=1024,  # Aumentado para diálogos más largos
                return_tensors=None
            )
            
            # Para entrenamiento causal, labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Aplicar tokenización
        print("  🔄 Tokenizando ejemplos...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=100,  # Procesar en lotes para eficiencia
            remove_columns=dataset.column_names,
            desc="Tokenizando"
        )
        
        print(f"  ✅ Dataset tokenizado: {len(tokenized_dataset)} ejemplos")
        
        # 4. Configurar entrenamiento
        print("\n⚙️ Configurando entrenamiento...")
        training_args = create_training_args_optimized()
        
        print(f"📁 Directorio de salida: {training_args.output_dir}")
        print(f"🔄 Épocas: {training_args.num_train_epochs}")
        print(f"📦 Batch size efectivo: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
        print(f"📈 Learning rate: {training_args.learning_rate}")
        print(f"💾 Gradient checkpointing: {training_args.gradient_checkpointing} (compatible con PEFT)")
        
        # 5. Configurar trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True
            )
        )
        
        # 6. Entrenar
        print("\n🚀 Iniciando entrenamiento DoRA...")
        print("=" * 70)
        print("💡 Configuración optimizada para RTX 3080 Ti")
        print("🔧 Solución PEFT + Gradient Checkpointing aplicada")
        print("💾 Memoria VRAM optimizada sin Flash Attention")
        
        trainer.train()
        
        # 7. Guardar modelo
        print("\n💾 Guardando modelo entrenado...")
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        
        # Guardar metadatos del entrenamiento
        metadata = {
            "model_name": "meta-llama/Llama-3.2-1B-Instruct",
            "technique": "DoRA (Decomposed Weight-based Low-Rank Adaptation)",
            "dataset": "empathy_dataset_smote_balanced.jsonl",
            "dataset_size": len(dataset),
            "training_time_hours": (time.time() - start_time) / 3600,
            "gpu": torch.cuda.get_device_name(0),
            "attention_implementation": "eager (standard, no Flash Attention)",
            "peft_solution": "prepare_model_for_kbit_training + enable_input_require_grads",
            "training_args": {
                "epochs": training_args.num_train_epochs,
                "batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation": training_args.gradient_accumulation_steps,
                "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
                "learning_rate": training_args.learning_rate,
                "rank": 32,
                "alpha": 64,
                "gradient_checkpointing": training_args.gradient_checkpointing
            },
            "timestamp": datetime.now().isoformat()
        }
        
        with open(f"{training_args.output_dir}/training_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Estadísticas finales
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print("=" * 70)
        print(f"⏱️ Tiempo total: {training_time/3600:.2f} horas")
        print(f"📁 Modelo guardado en: {training_args.output_dir}")
        print(f"📊 Ejemplos entrenados: {len(dataset):,}")
        print(f"🎯 Técnica: DoRA (Decomposed Weight-based Low-Rank Adaptation)")
        print(f"🧠 Capacidades: Empatía + Detección de Crisis + Análisis Multimodal")
        print(f"⚡ Implementación: Estándar (sin Flash Attention)")
        print(f"🔧 Solución: PEFT + Gradient Checkpointing compatible")
        
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Limpiar memoria GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()