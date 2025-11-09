#!/usr/bin/env python3
"""
Fusión del modelo de empatía entrenado con DoRA
Combina adaptadores DoRA con modelo base para crear modelo unificado
"""

import torch
import os
import glob
import json
from datetime import datetime

# Configuración CUDA para RTX 3080 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def find_latest_model():
    """Encontrar el modelo entrenado más reciente"""
    
    model_dirs = glob.glob("./empathy-llama-dora-smote_*")
    
    if not model_dirs:
        raise FileNotFoundError("No se encontró modelo entrenado. Ejecuta 1_train_dora_empathy_smote.py primero.")
    
    # Ordenar por timestamp y tomar el más reciente
    latest_model = sorted(model_dirs)[-1]
    print(f"📁 Modelo más reciente encontrado: {latest_model}")
    
    return latest_model

def load_training_metadata(model_path):
    """Cargar metadatos del entrenamiento"""
    
    metadata_file = os.path.join(model_path, "training_metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    else:
        return None

def fuse_empathy_dora_model(model_path):
    """Fusionar adaptadores DoRA con modelo base"""
    
    print("🔄 Fusionando modelo de empatía con DoRA...")
    print(f"📂 Modelo fuente: {model_path}")
    
    # Cargar metadatos si existen
    metadata = load_training_metadata(model_path)
    if metadata:
        print(f"📊 Dataset usado: {metadata.get('dataset_size', 'N/A')} ejemplos")
        print(f"⏱️ Tiempo de entrenamiento: {metadata.get('training_time_hours', 'N/A'):.2f} horas")
        print(f"🎯 Técnica: {metadata.get('technique', 'DoRA')}")
    
    # 1. Cargar modelo base
    print("\n📦 Cargando modelo base Llama 3.2 1B...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    print(f"✅ Modelo base cargado en: {base_model.device}")
    print(f"💾 Memoria GPU inicial: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 2. Cargar tokenizer
    print("📝 Cargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # 3. Cargar adaptadores DoRA
    print("⚡ Cargando adaptadores DoRA...")
    model_with_dora = PeftModel.from_pretrained(base_model, model_path)
    
    print(f"💾 Memoria GPU con DoRA: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 4. FUSIONAR (merge_and_unload)
    print("🔗 Fusionando adaptadores DoRA con modelo base...")
    print("   ⏳ Este proceso puede tomar varios minutos...")
    
    fused_model = model_with_dora.merge_and_unload()
    
    print("✅ Fusión completada exitosamente!")
    print(f"💾 Memoria GPU post-fusión: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # 5. Crear directorio de salida con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./empathy-fused-model_{timestamp}"
    
    print(f"\n💾 Guardando modelo fusionado en {output_dir}...")
    
    # Guardar modelo fusionado
    fused_model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size="2GB"  # Dividir en chunks para eficiencia
    )
    
    # Guardar tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Guardar metadatos de fusión
    fusion_metadata = {
        "source_model": model_path,
        "base_model": "meta-llama/Llama-3.2-1B-Instruct",
        "technique": "DoRA (Decomposed Weight-based Low-Rank Adaptation)",
        "fusion_timestamp": datetime.now().isoformat(),
        "capabilities": [
            "empathy_detection",
            "crisis_detection", 
            "multimodal_text_analysis",
            "spanish_peru_localization"
        ],
        "training_metadata": metadata
    }
    
    with open(f"{output_dir}/fusion_metadata.json", 'w', encoding='utf-8') as f:
        json.dump(fusion_metadata, f, indent=2, ensure_ascii=False)
    
    print("✅ Modelo y metadatos guardados exitosamente!")
    
    return fused_model, tokenizer, output_dir

def test_fused_model(model, tokenizer):
    """Realizar pruebas rápidas del modelo fusionado"""
    
    print("\n🧠 Pruebas rápidas del modelo fusionado:")
    print("-" * 50)
    
    test_cases = [
        "Hola, ¿cómo estás?",
        "Me siento muy triste hoy 😔",
        "No puedo más, quiero que todo termine",
        "Tengo ansiedad por mi examen de mañana"
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n🔸 Prueba {i}: {test_input}")
        
        # Formatear entrada
        messages = [{"role": "user", "content": test_input}]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenizar
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        # Generar respuesta
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decodificar respuesta
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        print(f"🤖 {response}")

def main():
    """Función principal de fusión"""
    
    print("🎯 Fusión del Modelo de Empatía con DoRA")
    print("=" * 60)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar CUDA
    if not torch.cuda.is_available():
        print("⚠️ CUDA no disponible. La fusión será más lenta en CPU.")
    else:
        print(f"✅ CUDA disponible: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    try:
        # Encontrar modelo más reciente
        model_path = find_latest_model()
        
        # Fusionar modelo
        fused_model, tokenizer, output_dir = fuse_empathy_dora_model(model_path)
        
        # Pruebas rápidas
        test_fused_model(fused_model, tokenizer)
        
        # Estadísticas finales
        print(f"\n🎉 ¡Fusión completada exitosamente!")
        print("=" * 60)
        print(f"📁 Modelo fusionado guardado en: {output_dir}")
        print(f"🧠 Capacidades: Empatía + Detección de Crisis + Análisis Multimodal")
        print(f"🌍 Idioma: Español (Perú)")
        print(f"⚡ Técnica: DoRA (Decomposed Weight-based Low-Rank Adaptation)")
        print(f"📊 Listo para conversión a GGUF y cuantización")
        
    except Exception as e:
        print(f"\n❌ Error durante la fusión: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Limpiar memoria GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"\n🧹 Memoria GPU liberada")

if __name__ == "__main__":
    main()
