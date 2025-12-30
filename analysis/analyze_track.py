"""
Script para analisar limites teóricos da pista.

Calcula:
- Distância mínima entre checkpoints
- Limite teórico de passos
- Compara com performance atual
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import config


def find_checkpoint_positions(track_string):
    """Encontra posição de cada checkpoint."""
    checkpoints = {'C': None, 'D': None, 'E': None, 'F': None}

    rows = track_string.split('\n')
    rows = [row.strip() for row in rows if row.strip()]

    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            if char in checkpoints:
                # Posição central do bloco em pixels
                x = col_idx * config.BLOCK_SIZE + config.BLOCK_SIZE // 2
                y = row_idx * config.BLOCK_SIZE + config.BLOCK_SIZE // 2
                checkpoints[char] = (x, y)

    return checkpoints


def calculate_straight_line_distances(checkpoints):
    """Calcula distâncias em linha reta entre checkpoints."""
    start = config.KART_INITIAL_POSITION

    # Ordem: Start -> C -> D -> E -> F
    points = [
        ('Start', start),
        ('C', checkpoints['C']),
        ('D', checkpoints['D']),
        ('E', checkpoints['E']),
        ('F', checkpoints['F'])
    ]

    total_distance = 0
    segments = []

    for i in range(len(points) - 1):
        name1, pos1 = points[i]
        name2, pos2 = points[i + 1]

        distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        total_distance += distance
        segments.append((name1, name2, distance))

    return total_distance, segments


def estimate_theoretical_minimum(total_distance):
    """Estima mínimo teórico de passos considerando física do kart."""
    # Velocidade média do kart
    # MAX_ACCELERATION = 0.25, então leva uns frames para acelerar
    # Velocidade típica em linha reta: ~5-8 pixels/frame

    # Estimativas:
    # - Linha reta perfeita: distance / 6 pixels por frame
    # - Com aceleração inicial: +10%
    # - Com curvas (virar leva tempo): +20%
    # - Ajustes finos: +5%

    ideal_straight = total_distance / 6.0  # Assumindo 6 px/frame em velocidade de cruzeiro
    with_acceleration = ideal_straight * 1.10  # Aceleração inicial
    with_turns = with_acceleration * 1.20  # Curvas
    with_fine_tuning = with_turns * 1.05  # Ajustes

    return {
        'ideal_straight': ideal_straight,
        'with_acceleration': with_acceleration,
        'with_turns': with_turns,
        'realistic_minimum': with_fine_tuning
    }


def main():
    print("\n" + "="*70)
    print("ANÁLISE DE LIMITES TEÓRICOS DA PISTA")
    print("="*70 + "\n")

    # Encontrar checkpoints
    checkpoints = find_checkpoint_positions(config.DEFAULT_TRACK)

    print("Posições dos Checkpoints:")
    for cp, pos in sorted(checkpoints.items()):
        print(f"  {cp}: {pos}")

    print(f"\nPosição inicial: {config.KART_INITIAL_POSITION}")

    # Calcular distâncias
    total_distance, segments = calculate_straight_line_distances(checkpoints)

    print("\n" + "-"*70)
    print("DISTÂNCIAS EM LINHA RETA (pixels):")
    print("-"*70)
    for name1, name2, dist in segments:
        print(f"  {name1} → {name2}: {dist:.1f} px")
    print(f"\n  TOTAL: {total_distance:.1f} pixels")

    # Estimar mínimos teóricos
    estimates = estimate_theoretical_minimum(total_distance)

    print("\n" + "-"*70)
    print("ESTIMATIVAS DE PASSOS MÍNIMOS:")
    print("-"*70)
    print(f"  Ideal (linha reta perfeita):        {estimates['ideal_straight']:.0f} passos")
    print(f"  Com aceleração inicial:             {estimates['with_acceleration']:.0f} passos")
    print(f"  Com curvas:                         {estimates['with_turns']:.0f} passos")
    print(f"  Mínimo realístico:                  {estimates['realistic_minimum']:.0f} passos")

    print("\n" + "-"*70)
    print("BENCHMARKS CONHECIDOS:")
    print("-"*70)
    print(f"  A* (original):                      ~464 passos")
    print(f"  A* (otimizado):                     ~440 passos")
    print(f"  Warm-start (imitação A*):           ~391 passos")
    print(f"  Neural Net (seu melhor):            ~271 passos  ⭐")

    # Análise
    current_best = 271
    theoretical_min = estimates['realistic_minimum']
    gap = current_best - theoretical_min
    efficiency = (theoretical_min / current_best) * 100

    print("\n" + "="*70)
    print("ANÁLISE:")
    print("="*70)
    print(f"  Gap para o mínimo teórico:          {gap:.0f} passos ({100-efficiency:.1f}% de overhead)")
    print(f"  Eficiência atual:                   {efficiency:.1f}%")

    if gap < 30:
        print("\n  🎯 PERTO DO LIMITE! Você já está muito próximo do ótimo teórico.")
        print("     Melhorias adicionais serão incrementais (<10 passos).")
    elif gap < 50:
        print("\n  ✅ MUITO BOM! Ainda tem espaço para otimizar (~20-30 passos).")
        print("     Continue treinando com população maior ou mais gerações.")
    else:
        print("\n  📈 TEM MARGEM! Ainda dá para melhorar bastante.")
        print("     Tente aumentar população e número de gerações.")

    # Convergência
    print("\n" + "-"*70)
    print("VERIFICAÇÃO DE CONVERGÊNCIA:")
    print("-"*70)
    print("  Se o modelo sempre faz 271 passos:")
    print("    - ✓ Convergiu (bom sinal de estabilidade)")
    print("    - Variação esperada: ±5-10 passos")
    print("\n  Para verificar se chegou no platô:")
    print("    - Treine mais 50-100 gerações")
    print("    - Se fitness não melhorar por 30+ gerações = PLATÔ")
    print("    - Se melhorar mesmo que pouco = continue!")

    print("\n" + "="*70)
    print("RECOMENDAÇÕES:")
    print("="*70)

    if current_best > theoretical_min + 50:
        print("  1. Continue treinando: python3 continue_training.py --model evolution_best --generations 100")
        print("  2. Aumente população: --population 100")
        print("  3. Reduza mutação inicial no config.py")
    elif current_best > theoretical_min + 30:
        print("  1. Treine mais: --generations 50-100")
        print("  2. População OK, mas pode tentar --population 75")
        print("  3. O modelo está bem otimizado!")
    else:
        print("  ⭐ EXCELENTE! Você já está quase no limite físico!")
        print("  1. Pequenos ajustes ainda possíveis (5-20 passos)")
        print("  2. Tente população maior (100+) com mutação mínima")
        print("  3. Ou aceite que 271 passos é IMPRESSIONANTE! 🏆")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
