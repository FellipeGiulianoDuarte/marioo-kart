"""
Script para visualizar métricas do treinamento evolutivo.

Uso:
    python plot_evolution.py                    # Plota o último treinamento
    python plot_evolution.py --file <arquivo>   # Plota arquivo específico
    python plot_evolution.py --all              # Plota todos os treinamentos
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import matplotlib.pyplot as plt
import numpy as np
import glob
import argparse


def plot_metrics(metrics_file):
    """Plota métricas de um arquivo."""
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    best_fitness = metrics['best_fitness']
    avg_fitness = metrics['avg_fitness']
    generations = list(range(len(best_fitness)))

    # Criar figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Fitness ao longo das gerações
    ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Melhor Fitness')
    ax1.plot(generations, avg_fitness, 'g--', linewidth=1.5, label='Fitness Médio')
    ax1.fill_between(generations, avg_fitness, alpha=0.3, color='green')
    ax1.set_xlabel('Geração', fontsize=12)
    ax1.set_ylabel('Fitness', fontsize=12)
    ax1.set_title('Evolução do Fitness', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Adicionar linha horizontal no benchmark do A* (se completar = 50000+)
    if max(best_fitness) > 40000:
        ax1.axhline(y=50000, color='r', linestyle=':', label='Baseline (Completar)')
        ax1.axhline(y=70000, color='orange', linestyle=':', label='Bateu A* (440 passos)')
        ax1.axhline(y=90000, color='purple', linestyle=':', label='Excelência (400 passos)')

    # Plot 2: Taxa de melhoria
    improvements = []
    for i in range(1, len(best_fitness)):
        improvement = best_fitness[i] - best_fitness[i-1]
        improvements.append(improvement)

    ax2.bar(range(1, len(improvements)+1), improvements, color=['green' if x > 0 else 'red' for x in improvements])
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Geração', fontsize=12)
    ax2.set_ylabel('Melhoria de Fitness', fontsize=12)
    ax2.set_title('Melhoria por Geração', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Informações gerais
    info_text = f"Gerações: {metrics['generations']}\n"
    info_text += f"População: {metrics['population_size']}\n"
    info_text += f"Melhor Fitness: {max(best_fitness):.1f}\n"
    info_text += f"Fitness Final: {best_fitness[-1]:.1f}"

    fig.text(0.02, 0.98, info_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.suptitle(f'Treinamento Evolutivo - {Path(metrics_file).stem}', fontsize=16, fontweight='bold', y=1.02)

    # Salvar figura
    output_file = metrics_file.replace('.json', '.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Gráfico salvo: {output_file}")

    plt.show()


def plot_comparison(metrics_files):
    """Compara múltiplos treinamentos."""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

    for idx, metrics_file in enumerate(metrics_files[:6]):  # Max 6 arquivos
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        best_fitness = metrics['best_fitness']
        generations = list(range(len(best_fitness)))

        label = f"Treino {idx+1} (Pop={metrics['population_size']})"
        ax.plot(generations, best_fitness, color=colors[idx], linewidth=2, label=label)

    ax.set_xlabel('Geração', fontsize=12)
    ax.set_ylabel('Melhor Fitness', fontsize=12)
    ax.set_title('Comparação de Treinamentos', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../metrics/evolution_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Comparação salva: metrics/evolution_comparison.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Visualizar métricas do treinamento evolutivo')
    parser.add_argument('--file', type=str, help='Arquivo de métricas específico')
    parser.add_argument('--all', action='store_true', help='Comparar todos os treinamentos')

    args = parser.parse_args()

    if args.all:
        # Comparar todos
        metrics_files = sorted(glob.glob('../metrics/evolution_metrics_*.json'))
        if not metrics_files:
            print("❌ Nenhum arquivo de métricas encontrado em metrics/")
            return

        print(f"Encontrados {len(metrics_files)} arquivos de métricas")
        plot_comparison(metrics_files)

    elif args.file:
        # Arquivo específico
        if not Path(args.file).exists():
            print(f"❌ Arquivo não encontrado: {args.file}")
            return

        plot_metrics(args.file)

    else:
        # Último arquivo
        metrics_files = sorted(glob.glob('../metrics/evolution_metrics_*.json'))
        if not metrics_files:
            print("❌ Nenhum arquivo de métricas encontrado em metrics/")
            print("Execute o treinamento primeiro:")
            print("  python continue_training.py --model warm_start --generations 20")
            return

        latest_file = metrics_files[-1]
        print(f"Plotando último treinamento: {latest_file}")
        plot_metrics(latest_file)


if __name__ == '__main__':
    main()
