#!/usr/bin/env python3
"""
Script para generar el gráfico de comparación de dimensiones (Baseline vs. Fine-tuned).

Autor: Manus AI
Fecha: 2025-01-03
Propósito: Visualizar la mejora en las 5 dimensiones terapéuticas evaluadas por el psicólogo.
"""

import matplotlib.pyplot as plt
import numpy as np

# Configuración de estilo profesional
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11

# Datos de evaluación (escala 1-5)
dimensiones = ['Empatía', 'Reconocimiento\nEmocional', 'Detección\nde Crisis', 'Respuesta\nApropiada', 'Calidad\nGeneral']
baseline = [3.13, 2.87, 1.29, 2.45, 2.61]
finetuned = [2.76, 3.42, 3.79, 3.18, 3.21]

# Calcular mejoras
mejoras = [(f - b) / b * 100 for b, f in zip(baseline, finetuned)]

# Crear figura con 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- SUBPLOT 1: Gráfico de barras agrupadas ---
x = np.arange(len(dimensiones))
width = 0.35

bars1 = ax1.bar(x - width/2, baseline, width, label='Baseline', color='#E74C3C', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, finetuned, width, label='Fine-tuned', color='#27AE60', alpha=0.8, edgecolor='black')

# Añadir valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9, weight='bold')

ax1.set_xlabel('Dimensión Terapéutica', fontsize=12, weight='bold')
ax1.set_ylabel('Puntuación Promedio (escala 1-5)', fontsize=12, weight='bold')
ax1.set_title('Comparación por Dimensión: Baseline vs. Fine-tuned', fontsize=13, weight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(dimensiones, fontsize=10)
ax1.set_ylim(0, 5)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# --- SUBPLOT 2: Gráfico de mejora porcentual ---
colors = ['#E74C3C' if m < 0 else '#27AE60' for m in mejoras]
bars3 = ax2.barh(dimensiones, mejoras, color=colors, alpha=0.8, edgecolor='black')

# Añadir valores al final de las barras
for i, (bar, mejora) in enumerate(zip(bars3, mejoras)):
    width = bar.get_width()
    ax2.text(width + (2 if width > 0 else -2), bar.get_y() + bar.get_height()/2.,
            f'{mejora:+.1f}%', ha='left' if width > 0 else 'right', va='center', 
            fontsize=10, weight='bold', color=colors[i])

ax2.set_xlabel('Mejora (%)', fontsize=12, weight='bold')
ax2.set_title('Mejora Porcentual por Dimensión', fontsize=13, weight='bold', pad=15)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.set_xlim(-15, 200)

# Añadir línea de promedio
promedio_mejora = np.mean(mejoras)
ax2.axvline(x=promedio_mejora, color='blue', linestyle=':', linewidth=2, label=f'Promedio: {promedio_mejora:+.1f}%')
ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)

plt.tight_layout()

# Guardar figura
output_path = '/home/ubuntu/dimension_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Gráfico de comparación por dimensión guardado en: {output_path}")

# Estadísticas
print(f"\n📊 Estadísticas de mejora por dimensión:")
for dim, b, f, m in zip(dimensiones, baseline, finetuned, mejoras):
    print(f"  {dim.replace(chr(10), ' ')}: {b:.2f} → {f:.2f} ({m:+.1f}%)")
print(f"\n📈 Mejora promedio: {promedio_mejora:+.1f}%")
print(f"🔝 Mayor mejora: Detección de Crisis (+{max(mejoras):.1f}%)")
print(f"🔻 Menor mejora: Empatía ({min(mejoras):.1f}%)")

plt.close()
