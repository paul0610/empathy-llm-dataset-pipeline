#!/usr/bin/env python3
"""
Cuantización del modelo de empatía GGUF a Q4_K_M
Optimizado para dispositivos móviles y React Native
"""

import subprocess
import os
import glob
import json
from datetime import datetime

def find_llama_quantize_executable():
    """Encontrar el ejecutable llama-quantize"""
    
    # Rutas comunes donde podría estar el ejecutable
    possible_paths = [
        "D:/proyectos/LLMS/DORA_FINETUNING/llama.cpp/build/bin/Release/llama-quantize.exe",  # Tu ruta original
        "./llama.cpp/build/bin/Release/llama-quantize.exe",
        "./llama.cpp/build/bin/llama-quantize.exe",
        "./llama.cpp/llama-quantize.exe",
        "../llama.cpp/build/bin/Release/llama-quantize.exe",
        "~/llama.cpp/build/bin/Release/llama-quantize.exe",
        "/usr/local/bin/llama-quantize",
        "./llama-quantize",
        "llama-quantize"  # Si está en PATH
    ]
    
    for path in possible_paths:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            print(f"✅ llama-quantize encontrado en: {expanded_path}")
            return expanded_path
    
    # Si no se encuentra, pedir al usuario
    print("⚠️ No se encontró llama-quantize automáticamente.")
    print("💡 Por favor, especifica la ruta completa del ejecutable llama-quantize:")
    print("   Ejemplo Windows: D:/proyectos/LLMS/DORA_FINETUNING/llama.cpp/build/bin/Release/llama-quantize.exe")
    print("   Ejemplo Linux/Mac: ~/llama.cpp/build/bin/llama-quantize")
    
    user_path = input("📁 Ruta de llama-quantize: ").strip()
    
    if os.path.exists(user_path):
        return user_path
    else:
        raise FileNotFoundError(f"No se encontró llama-quantize en {user_path}")

def find_latest_gguf_model():
    """Encontrar el modelo GGUF más reciente"""
    
    # Buscar directorios de GGUF
    gguf_dirs = glob.glob("./empathy-gguf-*")
    
    if not gguf_dirs:
        raise FileNotFoundError("No se encontró directorio de GGUF. Ejecuta 4_convert_empathy_to_gguf.py primero.")
    
    # Buscar archivos GGUF en el directorio más reciente
    latest_dir = sorted(gguf_dirs)[-1]
    gguf_files = glob.glob(f"{latest_dir}/*.gguf")
    
    if not gguf_files:
        raise FileNotFoundError(f"No se encontraron archivos GGUF en {latest_dir}")
    
    # Preferir Q8_0, luego F16
    q8_files = [f for f in gguf_files if "q8_0" in f.lower()]
    f16_files = [f for f in gguf_files if "f16" in f.lower()]
    
    if q8_files:
        input_file = q8_files[0]
        print(f"📁 Usando modelo Q8_0: {input_file}")
    elif f16_files:
        input_file = f16_files[0]
        print(f"📁 Usando modelo F16: {input_file}")
    else:
        input_file = gguf_files[0]
        print(f"📁 Usando modelo: {input_file}")
    
    return input_file, latest_dir

def load_gguf_metadata(gguf_dir):
    """Cargar metadatos del GGUF si existen"""
    
    metadata_file = os.path.join(gguf_dir, "gguf_metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        return metadata
    else:
        return None

def quantize_to_q4_k_m(input_file, quantize_exe):
    """Cuantizar modelo GGUF a Q4_K_M"""
    
    print("🔄 Cuantizando modelo de empatía a Q4_K_M...")
    print(f"📂 Archivo de entrada: {input_file}")
    
    # Crear nombre de archivo de salida
    input_dir = os.path.dirname(input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(input_dir, f"empathy-llama-q4_k_m_{timestamp}.gguf")
    
    # Verificar archivo de entrada
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Archivo de entrada no encontrado: {input_file}")
    
    input_size_mb = os.path.getsize(input_file) / (1024*1024)
    print(f"📊 Tamaño de entrada: {input_size_mb:.1f} MB")
    
    # Comando de cuantización
    cmd = [quantize_exe, input_file, output_file, "Q4_K_M"]
    
    print(f"🚀 Ejecutando cuantización...")
    print(f"   🔧 Formato objetivo: Q4_K_M (4-bit, optimizado para móvil)")
    print(f"   📱 Salida: {os.path.basename(output_file)}")
    
    try:
        # Ejecutar cuantización
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        print("\n📋 Proceso de cuantización:")
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("\n⚠️ Mensajes del proceso:")
            print(result.stderr)
        
        if result.returncode == 0 and os.path.exists(output_file):
            # Calcular estadísticas
            output_size_mb = os.path.getsize(output_file) / (1024*1024)
            reduction = ((input_size_mb - output_size_mb) / input_size_mb) * 100
            
            print("\n🎉 ¡CUANTIZACIÓN EXITOSA!")
            print("=" * 50)
            print(f"📊 Modelo original: {input_size_mb:.1f} MB")
            print(f"📱 Modelo Q4_K_M: {output_size_mb:.1f} MB")
            print(f"💾 Reducción de tamaño: {reduction:.1f}%")
            print(f"⚡ Velocidad esperada: ~2-3x más rápido")
            print(f"🧠 Calidad: ~95% del modelo original")
            
            # Crear metadatos de cuantización
            quantization_metadata = {
                "input_file": input_file,
                "output_file": output_file,
                "quantization_type": "Q4_K_M",
                "input_size_mb": round(input_size_mb, 1),
                "output_size_mb": round(output_size_mb, 1),
                "size_reduction_percent": round(reduction, 1),
                "quantization_timestamp": datetime.now().isoformat(),
                "target_platforms": [
                    "Android (React Native)",
                    "iOS (React Native)", 
                    "llama.cpp mobile",
                    "Edge devices"
                ],
                "performance": {
                    "memory_usage": "Low (~600-800 MB)",
                    "inference_speed": "Fast (2-3x faster than Q8_0)",
                    "quality_retention": "~95% of original model",
                    "recommended_for": "Mobile deployment"
                },
                "capabilities": [
                    "empathy_detection",
                    "crisis_detection",
                    "multimodal_text_analysis",
                    "spanish_peru_localization"
                ]
            }
            
            metadata_file = os.path.join(input_dir, f"q4_quantization_metadata_{timestamp}.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(quantization_metadata, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Metadatos guardados en: {os.path.basename(metadata_file)}")
            
            print(f"\n🚀 ¡MODELO LISTO PARA REACT NATIVE!")
            print("=" * 50)
            print(f"📁 Archivo final: {output_file}")
            print(f"📱 Tamaño optimizado: {output_size_mb:.1f} MB")
            print(f"🎯 Perfecto para dispositivos móviles")
            print(f"🌍 Idioma: Español (Perú)")
            print(f"🧠 Capacidades: Empatía + Crisis + Multimodal")
            
            return output_file, quantization_metadata
            
        else:
            print(f"\n❌ Error en cuantización (código de salida: {result.returncode})")
            return None, None
            
    except Exception as e:
        print(f"❌ Error durante la cuantización: {e}")
        return None, None

def test_quantized_model_info(output_file):
    """Mostrar información del modelo cuantizado"""
    
    if not os.path.exists(output_file):
        return
    
    print(f"\n📋 INFORMACIÓN DEL MODELO CUANTIZADO:")
    print("-" * 40)
    
    size_mb = os.path.getsize(output_file) / (1024*1024)
    print(f"📦 Archivo: {os.path.basename(output_file)}")
    print(f"📊 Tamaño: {size_mb:.1f} MB")
    print(f"🔧 Formato: Q4_K_M GGUF")
    print(f"📱 Optimizado para: Dispositivos móviles")
    print(f"⚡ Velocidad: Rápida (4-bit quantization)")
    print(f"🧠 Calidad: Alta (~95% retención)")
    
    # Estimaciones de rendimiento
    print(f"\n⚡ ESTIMACIONES DE RENDIMIENTO:")
    print(f"  💾 RAM requerida: ~600-800 MB")
    print(f"  🚀 Velocidad de inferencia: 2-3x más rápido que Q8_0")
    print(f"  🔋 Consumo de batería: Optimizado para móvil")
    print(f"  📱 Compatible con: React Native + llama.cpp")

def main():
    """Función principal de cuantización"""
    
    print("🎯 Cuantización del Modelo de Empatía a Q4_K_M")
    print("=" * 60)
    print(f"🕐 Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Encontrar ejecutable llama-quantize
        print("\n🔧 Configurando llama-quantize...")
        quantize_exe = find_llama_quantize_executable()
        
        # 2. Encontrar modelo GGUF más reciente
        print(f"\n📂 Buscando modelo GGUF...")
        input_file, gguf_dir = find_latest_gguf_model()
        
        # 3. Cargar metadatos si existen
        metadata = load_gguf_metadata(gguf_dir)
        if metadata:
            print(f"🧠 Capacidades: {', '.join(metadata.get('capabilities', []))}")
            print(f"📊 Tamaño original: {metadata.get('size_mb', 'N/A')} MB")
        
        # 4. Ejecutar cuantización
        print(f"\n🚀 Iniciando cuantización a Q4_K_M...")
        output_file, quant_metadata = quantize_to_q4_k_m(input_file, quantize_exe)
        
        if output_file and os.path.exists(output_file):
            # 5. Mostrar información del modelo final
            test_quantized_model_info(output_file)
            
            print(f"\n📋 PRÓXIMOS PASOS:")
            print(f"  1. Copiar {os.path.basename(output_file)} a tu proyecto React Native")
            print(f"  2. Integrar con llama.cpp en tu aplicación móvil")
            print(f"  3. Configurar la base de datos vectorial local")
            print(f"  4. Probar detección de empatía y crisis en dispositivo")
            
        else:
            print("❌ La cuantización falló")
        
    except Exception as e:
        print(f"\n❌ Error durante la cuantización: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
