#!/usr/bin/env python3
"""
Script para generar el gráfico de trade-off entre tamaño del modelo y calidad de respuestas.

Autor: Manus AI
Fecha: 2025-01-03
Propósito: Visualizar el balance entre compresión y preservación de calidad en diferentes
          formatos de cuantización (FP16, Q8_0, Q4_K_M) para el TFM.
"""

import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Datos de los formatos
formatos = ['FP16', 'Q8_0', 'Q4_K_M']
tamanos_mb = [2400, 1300, 800]
calidades = [4.6, 4.5, 4.3]  # Promedio de las 3 dimensiones

# Colores para cada formato
colores = ['#2E86AB', '#A23B72', '#F18F01']  # Azul, Morado, Naranja

# Crear figura
fig, ax = plt.subplots(figsize=(10, 7))

# Scatter plot con tamaños proporcionales
sizes = [300, 250, 200]  # Tamaños de los puntos
for i, (fmt, tam, cal, col, size) in enumerate(zip(formatos, tamanos_mb, calidades, colores, sizes)):
    ax.scatter(tam, cal, s=size, c=col, alpha=0.7, edgecolors='black', linewidth=1.5, label=fmt, zorder=3)
    
    # Añadir etiquetas con valores
    offset_x = 50 if i == 0 else (-50 if i == 2 else 0)
    offset_y = 0.05 if i == 1 else -0.08
    ax.annotate(f'{fmt}\n({tam} MB, {cal}/5.0)', 
                xy=(tam, cal), 
                xytext=(tam + offset_x, cal + offset_y),
                fontsize=9,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=col, alpha=0.2, edgecolor=col),
                zorder=4)

# Línea de tendencia (curva de Pareto)
z = np.polyfit(tamanos_mb, calidades, 2)
p = np.poly1d(z)
x_smooth = np.linspace(min(tamanos_mb), max(tamanos_mb), 100)
y_smooth = p(x_smooth)
ax.plot(x_smooth, y_smooth, '--', color='gray', alpha=0.5, linewidth=1.5, label='Tendencia', zorder=2)

# Destacar Q4_K_M como óptimo
ax.annotate('⭐ Óptimo\n(Balance tamaño-calidad)', 
            xy=(800, 4.3), 
            xytext=(1000, 4.0),
            fontsize=10,
            ha='center',
            color='#F18F01',
            weight='bold',
            arrowprops=dict(arrowstyle='->', color='#F18F01', lw=2),
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#F18F01', alpha=0.15, edgecolor='#F18F01', linewidth=2),
            zorder=5)

# Configuración de ejes
ax.set_xlabel('Tamaño del Modelo (MB)', fontsize=12, weight='bold')
ax.set_ylabel('Calidad Promedio (escala 1-5)', fontsize=12, weight='bold')
ax.set_title('Trade-off entre Tamaño del Modelo y Calidad de Respuestas\nFormatos de Cuantización: FP16, Q8_0, Q4_K_M', 
             fontsize=13, weight='bold', pad=15)

# Límites de ejes
ax.set_xlim(600, 2600)
ax.set_ylim(4.0, 4.8)

# Grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)

# Leyenda
ax.legend(loc='upper right', framealpha=0.9, edgecolor='black', fancybox=True)

# Añadir anotaciones de reducción de tamaño
ax.annotate('', xy=(800, 4.75), xytext=(2400, 4.75),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
ax.text(1600, 4.77, '67% reducción', ha='center', fontsize=9, color='red', weight='bold')

# Añadir anotaciones de degradación de calidad
ax.annotate('', xy=(2500, 4.3), xytext=(2500, 4.6),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.5))
ax.text(2550, 4.45, '6.5%\ndegradación', ha='left', fontsize=9, color='blue', weight='bold', va='center')

# Ajustar layout
plt.tight_layout()

# Guardar figura
output_path = '/home/ubuntu/tradeoff_tamano_calidad.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico de trade-off guardado en: {output_path}")

# Estadísticas
print(f"\n📊 Estadísticas del trade-off:")
print(f"  FP16:    {tamanos_mb[0]} MB, Calidad {calidades[0]}/5.0")
print(f"  Q8_0:    {tamanos_mb[1]} MB, Calidad {calidades[1]}/5.0 (Reducción: {100*(1-tamanos_mb[1]/tamanos_mb[0]):.1f}%, Degradación: {100*(1-calidades[1]/calidades[0]):.1f}%)")
print(f"  Q4_K_M:  {tamanos_mb[2]} MB, Calidad {calidades[2]}/5.0 (Reducción: {100*(1-tamanos_mb[2]/tamanos_mb[0]):.1f}%, Degradación: {100*(1-calidades[2]/calidades[0]):.1f}%)")
print(f"\n⭐ Q4_K_M ofrece el mejor balance: 67% reducción con solo 6.5% degradación")

plt.close()
