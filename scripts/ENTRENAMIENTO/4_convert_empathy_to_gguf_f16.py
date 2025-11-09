#!/usr/bin/env python3
"""
Conversión del modelo de empatía fusionado a formato GGUF F16
Optimizado para llama.cpp y posterior cuantización a Q4_K_M
VERSIÓN F16: Para máxima calidad en cuantización posterior
"""

import subprocess
import os
import glob
import json
from datetime import datetime

def find_latest_fused_model():
    """Encontrar el modelo fusionado más reciente"""
    
    model_dirs = glob.glob("./empathy-fused-model_*")
    
    if not model_dirs:
        raise FileNotFoundError("No se encontró modelo fusionado. Ejecuta 3_fusion_empathy_model.py primero.")
    
    # Ordenar por timestamp y tomar el más reciente
    latest_model = sorted(model_dirs)[-1]
    print(f"📁 Modelo fusionado más reciente: {latest_model}")
    
    return latest_model

def load_fusion_metadata(model_path):
    """Cargar metadatos de la fusión"""
    
    metadata_file = os.path.join(model_path, "fusion_metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    else:
        return None

def setup_llama_cpp_path():
    """Configurar ruta de llama.cpp - ajustar según tu instalación"""
    
    # Rutas comunes donde podría estar llama.cpp
    possible_paths = [
        "./llama.cpp",
        "../llama.cpp",
        "~/llama.cpp",
        "/usr/local/llama.cpp",
        "/workspace/llama.cpp"
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        convert_script = os.path.join(expanded_path, "convert_hf_to_gguf.py")
        
        if os.path.exists(convert_script):
            print(f"✅ llama.cpp encontrado en: {expanded_path}")
            return expanded_path
    
    # Si no se encuentra, pedir al usuario
    print("⚠️ No se encontró llama.cpp automáticamente.")
    print("💡 Por favor, especifica la ruta donde tienes clonado llama.cpp:")
    print("   Ejemplo: /workspace/llama.cpp")
    
    user_path = input("📁 Ruta de llama.cpp: ").strip()
    
    if os.path.exists(os.path.join(user_path, "convert_hf_to_gguf.py")):
        return user_path
    else:
        raise FileNotFoundError(f"No se encontró convert_hf_to_gguf.py en {user_path}")

def convert_to_gguf_f16(model_dir, llama_cpp_path):
    """Convertir modelo fusionado a F16 GGUF (máxima calidad)"""
    
    print("🔄 Convirtiendo modelo de empatía a F16 GGUF...")
    print(f"📂 Modelo fuente: {model_dir}")
    print("💎 Formato F16: Máxima calidad para cuantización posterior")
    
    # Crear directorio de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gguf_dir = f"./empathy-gguf-f16_{timestamp}"
    os.makedirs(gguf_dir, exist_ok=True)
    
    # Archivo de salida F16
    output_file = f"{gguf_dir}/empathy-llama-f16.gguf"
    
    # Script de conversión
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    
    # Comando de conversión a F16
    cmd = [
        "python", convert_script,
        model_dir,                    # Directorio del modelo fusionado
        "--outfile", output_file,     # Archivo de salida específico
        "--outtype", "f16"           # Conversión a F16 (16-bit float, máxima calidad)
    ]
    
    print(f"🚀 Ejecutando conversión: {' '.join(cmd)}")
    print("📋 Método: Conversión directa HF → F16 (preserva calidad completa)")
    
    try:
        # Ejecutar conversión
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        print("\n📋 Salida del proceso de conversión:")
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Mensajes de advertencia:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("\n✅ Conversión a F16 exitosa!")
            
            # Verificar archivo creado
            if os.path.exists(output_file):
                size_mb = os.path.getsize(output_file) / (1024*1024)
                print(f"📦 Archivo creado: empathy-llama-f16.gguf")
                print(f"📊 Tamaño: {size_mb:.1f} MB")
                print(f"🎯 Tamaño esperado para F16: ~2400-2600 MB")
                
                # Crear metadatos del GGUF
                gguf_metadata = {
                    "source_model": model_dir,
                    "output_file": output_file,
                    "format": "GGUF F16",
                    "size_mb": round(size_mb, 1),
                    "conversion_timestamp": datetime.now().isoformat(),
                    "capabilities": [
                        "empathy_detection",
                        "crisis_detection",
                        "multimodal_text_analysis",
                        "spanish_peru_localization",
                        "mobile_optimized"
                    ],
                    "target_platforms": [
                        "llama.cpp",
                        "Android (via llama.cpp)",
                        "iOS (via llama.cpp)",
                        "React Native"
                    ],
                    "quantization": {
                        "type": "F16",
                        "bits_per_weight": 16,
                        "quality": "Maximum (100% quality retention)",
                        "speed": "Baseline",
                        "memory_usage": "High",
                        "notes": "Base format for optimal quantization to Q4_K_M, Q5_K_M, Q8_0, etc."
                    }
                }
                
                metadata_file = f"{gguf_dir}/gguf_metadata.json"
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(gguf_metadata, f, indent=2, ensure_ascii=False)
                
                print(f"📄 Metadatos guardados en: {metadata_file}")
                
                return output_file, gguf_dir
                
            else:
                print("❌ Archivo GGUF no encontrado después de la conversión")
                return None, None
                
        else:
            print(f"❌ Error en conversión (código de salida: {result.returncode})")
            return None, None
            
    except Exception as e:
        print(f"❌ Error durante la conversión: {e}")
        return None, None

def main():
    """Función principal de conversión"""
    
    print("🎯 Conversión del Modelo de Empatía a GGUF F16")
    print("=" * 60)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💎 Formato F16: Máxima calidad para cuantización posterior")
    
    try:
        # 1. Encontrar modelo fusionado más reciente
        model_dir = find_latest_fused_model()
        
        # 2. Cargar metadatos si existen
        metadata = load_fusion_metadata(model_dir)
        if metadata:
            print(f"🧠 Capacidades: {', '.join(metadata.get('capabilities', []))}")
            print(f"⚡ Técnica: {metadata.get('technique', 'DoRA')}")
        
        # 3. Configurar llama.cpp
        print(f"\n🔧 Configurando llama.cpp...")
        llama_cpp_path = setup_llama_cpp_path()
        
        # 4. Conversión a F16
        print(f"\n🚀 Iniciando conversión a GGUF F16...")
        output_file, gguf_dir = convert_to_gguf_f16(model_dir, llama_cpp_path)
        
        if output_file and os.path.exists(output_file):
            print(f"\n🎉 ¡Conversión completada exitosamente!")
            print("=" * 60)
            print(f"📁 Archivo GGUF: {output_file}")
            print(f"📂 Directorio: {gguf_dir}")
            print(f"🎯 Formato: F16 (16-bit float, máxima calidad)")
            print(f"📱 Optimizado para: Cuantización posterior")
            print(f"🌍 Idioma: Español (Perú)")
            print(f"🧠 Capacidades: Empatía + Crisis + Multimodal")
            print(f"\n📋 Próximo paso: Ejecutar 5_quantize_empathy_q4.py para Q4_K_M")
            print(f"💡 Nota: El script 5 detectará automáticamente el archivo F16")
            
        else:
            print(f"\n❌ Conversión F16 falló")
        
    except Exception as e:
        print(f"\n❌ Error durante la conversión: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

