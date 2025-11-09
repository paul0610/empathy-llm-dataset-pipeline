#!/usr/bin/env python3
"""
Script para generar gráfico de tasa de detección por categoría adversarial.

Autor: Manus AI
Fecha: 2025-01-03
Propósito: Visualizar el rendimiento del modelo en casos adversariales.
"""

import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Datos de detección por categoría adversarial
categorias = ['Sarcasmo', 'Ambigüedad', 'Cambios\nBruscos', 'Negación']
deteccion_correcta = [78, 85, 68, 88]  # Porcentajes
respuesta_apropiada = [71, 78, 58, 82]  # Porcentajes

# Crear figura
fig, ax = plt.subplots(figsize=(12, 7))

# Posiciones de las barras
x = np.arange(len(categorias))
width = 0.35

# Barras
bars1 = ax.bar(x - width/2, deteccion_correcta, width, label='Detección Correcta', 
               color='#3498DB', edgecolor='black', linewidth=1.2)
bars2 = ax.bar(x + width/2, respuesta_apropiada, width, label='Respuesta Apropiada', 
               color='#2ECC71', edgecolor='black', linewidth=1.2)

# Añadir valores en las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%',
                ha='center', va='bottom', fontsize=10, weight='bold')

# Línea de umbral aceptable (70%)
ax.axhline(y=70, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Umbral Aceptable (70%)')

# Configuración de ejes
ax.set_xlabel('Categoría Adversarial', fontsize=12, weight='bold')
ax.set_ylabel('Tasa de Éxito (%)', fontsize=12, weight='bold')
ax.set_title('Rendimiento del Modelo en Casos Adversariales\n(Detección Correcta vs. Respuesta Apropiada)', 
             fontsize=13, weight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categorias, fontsize=11)
ax.set_ylim(0, 100)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

# Leyenda
ax.legend(loc='lower right', fontsize=10, framealpha=0.9, edgecolor='black')

# Añadir anotaciones
ax.annotate('Mejor rendimiento\n(88% detección)', xy=(3, 88), xytext=(3.3, 92),
            fontsize=9, color='darkgreen', weight='bold',
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', edgecolor='darkgreen', alpha=0.7))

ax.annotate('Peor rendimiento\n(58% respuesta)', xy=(2, 58), xytext=(1.2, 50),
            fontsize=9, color='darkred', weight='bold',
            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', edgecolor='darkred', alpha=0.7))

plt.tight_layout()

# Guardar figura
output_path = '/home/ubuntu/tasa_deteccion_adversarial.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico de tasa de detección guardado en: {output_path}")

# Estadísticas
print(f"\n📊 Estadísticas de rendimiento adversarial:")
print(f"  Promedio Detección Correcta: {np.mean(deteccion_correcta):.1f}%")
print(f"  Promedio Respuesta Apropiada: {np.mean(respuesta_apropiada):.1f}%")
print(f"  Mejor categoría (Detección): {categorias[np.argmax(deteccion_correcta)]} ({max(deteccion_correcta)}%)")
print(f"  Peor categoría (Detección): {categorias[np.argmin(deteccion_correcta)]} ({min(deteccion_correcta)}%)")
print(f"  Mejor categoría (Respuesta): {categorias[np.argmax(respuesta_apropiada)]} ({max(respuesta_apropiada)}%)")
print(f"  Peor categoría (Respuesta): {categorias[np.argmin(respuesta_apropiada)]} ({min(respuesta_apropiada)}%)")

# Análisis de categorías bajo umbral
bajo_umbral_deteccion = [cat for i, cat in enumerate(categorias) if deteccion_correcta[i] < 70]
bajo_umbral_respuesta = [cat for i, cat in enumerate(categorias) if respuesta_apropiada[i] < 70]

print(f"\n⚠️ Categorías bajo umbral aceptable (70%):")
if bajo_umbral_deteccion:
    print(f"  Detección: {', '.join(bajo_umbral_deteccion)}")
else:
    print(f"  Detección: Ninguna")
    
if bajo_umbral_respuesta:
    print(f"  Respuesta: {', '.join(bajo_umbral_respuesta)}")
else:
    print(f"  Respuesta: Ninguna")

plt.close()
