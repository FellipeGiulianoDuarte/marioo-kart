"""
Script para continuar treinamento evolutivo a partir de um modelo salvo.

Uso:
    python continue_training.py                          # Continua do melhor modelo
    python continue_training.py --model evolution_gen50  # Continua de um modelo específico
    python continue_training.py --generations 100        # Define número de gerações
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import config
from train_evolution import EvolutionaryTrainer, EvolutionaryKart
from rl.rl_controller import RLController


def main():
    parser = argparse.ArgumentParser(description='Continuar treinamento evolutivo')
    parser.add_argument('--track', type=str, default='pista1',
                        help=f'Pista para treinar (default: pista1, opções: {list(config.TRACKS.keys())})')
    parser.add_argument('--model', type=str, default=None,
                        help='Nome do modelo (sem .pth). Se não especificado, usa o best da pista.')
    parser.add_argument('--generations', type=int, default=100,
                        help='Número de gerações adicionais')
    parser.add_argument('--population', type=int, default=config.DEFAULT_POPULATION_SIZE,
                        help='Tamanho da população')

    args = parser.parse_args()

    # Validar pista
    if args.track not in config.TRACKS:
        print(f"❌ Erro: Pista '{args.track}' não existe.")
        print(f"   Pistas disponíveis: {list(config.TRACKS.keys())}")
        return

    # Definir pista ativa
    config.set_active_track(args.track)
    track_config = config.get_active_track()

    if track_config['track_string'] is None:
        print(f"❌ Erro: Pista '{args.track}' não está definida ainda.")
        return

    # Definir modelo (usar best da pista se não especificado)
    if args.model is None:
        model_name = track_config['best_model'].replace('.pth', '')
    else:
        model_name = args.model

    # Criar trainer
    trainer = EvolutionaryTrainer(
        track_config['track_string'],
        track_config['initial_position'],
        track_config['initial_angle'],
        population_size=args.population
    )

    # Carregar modelo salvo
    model_path = f"{config.MODEL_DIR}/{model_name}.pth"
    print(f"\n{'='*60}")
    print(f"CONTINUANDO TREINAMENTO - {track_config['name']}")
    print(f"{'='*60}")
    print(f"Pista: {args.track}")
    print(f"Modelo base: {model_path}")
    print(f"População: {args.population} karts")
    print(f"Gerações adicionais: {args.generations}")
    print(f"{'='*60}\n")

    try:
        # Carregar o melhor modelo para a população
        best_controller = RLController(state_size=12, action_size=7)
        best_controller.load(model_path)
        best_controller.epsilon = 0.0

        print(f"✓ Modelo carregado com sucesso!")
        print(f"  Inicializando população a partir do modelo...")

        # Substituir população inicial com clones mutados do melhor modelo
        trainer.population = []

        # Primeiro kart: cópia exata (elite)
        elite_controller = RLController(state_size=12, action_size=7)
        elite_controller.policy_net.load_state_dict(best_controller.policy_net.state_dict())
        elite_controller.epsilon = 0.0
        trainer.population.append(EvolutionaryKart(elite_controller, 0))

        # Resto: clones com mutações crescentes
        for i in range(1, args.population):
            # Criar clone
            clone_controller = RLController(state_size=12, action_size=7)
            clone_controller.policy_net.load_state_dict(best_controller.policy_net.state_dict())
            clone_controller.epsilon = 0.0

            # Aplicar mutação (mais mutação = mais diversidade)
            # Primeiros são quase idênticos, últimos têm mais variação
            mutation_strength = 0.05 + (i / args.population) * 0.25
            trainer._mutate_network(clone_controller.policy_net, mutation_strength)

            trainer.population.append(EvolutionaryKart(clone_controller, i))

        print(f"✓ População inicializada com variações do modelo base")
        print(f"  Mutação: 5% a 30% (diversidade genética)")
        print(f"\nIniciando evolução...\n")

    except Exception as e:
        print(f"\n❌ Erro ao carregar modelo: {e}")
        print(f"   Iniciando com população aleatória...")

    # Treinar
    trainer.train(num_generations=args.generations)


if __name__ == '__main__':
    main()
