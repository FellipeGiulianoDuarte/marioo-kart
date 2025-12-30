"""
Script para criar modelo warm-start através de Imitation Learning do A*.

Roda o A* várias vezes, coleta (estado, ação) pares,
e treina uma rede neural para imitar o A*.

Esse modelo serve como ponto de partida para o treinamento evolutivo.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pygame
import numpy as np
from collections import deque
import config
from game.entities.track import Track
from game.entities.kart import Kart
from game.controllers.ai import AI
from rl.rl_controller import RLController


def collect_expert_demonstrations(num_episodes=50):
    """Coleta demonstrações do A* (expert)."""
    print(f"\n{'='*60}")
    print("COLETANDO DEMONSTRAÇÕES DO A* (EXPERT)")
    print(f"{'='*60}\n")

    demonstrations = []  # Lista de (state, action)

    # Criar A* controller
    ai_controller = AI()

    for episode in range(num_episodes):
        print(f"Episódio {episode+1}/{num_episodes}...", end=" ")

        # Criar kart e track
        kart = Kart(ai_controller)
        track = Track(config.DEFAULT_TRACK, config.KART_INITIAL_POSITION, config.KART_INITIAL_ANGLE)
        track.add_kart(kart)

        # Resetar kart
        kart.reset(config.KART_INITIAL_POSITION, config.KART_INITIAL_ANGLE)

        # Inicializar pygame (necessário mas não mostrar tela)
        pygame.init()
        screen = pygame.display.set_mode((track.width, track.height))

        # Rodar episódio
        steps = 0
        max_steps = 1000

        while not kart.has_finished and steps < max_steps:
            # Obter ação do A*
            keys = ai_controller.move(config.DEFAULT_TRACK)

            # Extrair ação como array binário
            action = np.array([
                1 if keys[pygame.K_UP] else 0,
                1 if keys[pygame.K_DOWN] else 0,
                1 if keys[pygame.K_LEFT] else 0,
                1 if keys[pygame.K_RIGHT] else 0
            ], dtype=np.float32)

            # Obter estado (usando o método do RL controller)
            dummy_rl = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
            dummy_rl.kart = kart
            state = dummy_rl.get_state(config.DEFAULT_TRACK)

            # Salvar (estado, ação)
            demonstrations.append((state, action))

            # Aplicar ação
            if keys[pygame.K_UP]:
                kart.forward()
            if keys[pygame.K_DOWN]:
                kart.backward()
            if keys[pygame.K_LEFT]:
                kart.turn_left()
            if keys[pygame.K_RIGHT]:
                kart.turn_right()

            # Atualizar kart
            kart.update_position(config.DEFAULT_TRACK, screen)

            steps += 1

        if kart.has_finished:
            print(f"✓ Completou em {steps} passos ({len(demonstrations)} amostras totais)")
        else:
            print(f"✗ Timeout após {steps} passos")

        pygame.quit()

    print(f"\n{'='*60}")
    print(f"COLETADAS {len(demonstrations)} DEMONSTRAÇÕES")
    print(f"{'='*60}\n")

    return demonstrations


def train_imitation_model(demonstrations, epochs=100):
    """Treina modelo para imitar demonstrações."""
    print(f"\n{'='*60}")
    print("TREINANDO MODELO POR IMIT ATION LEARNING")
    print(f"{'='*60}\n")

    # Criar modelo
    model = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)

    # Converter demonstrações para tensors
    states = torch.tensor([s for s, a in demonstrations], dtype=torch.float32)

    # Converter ações para índices (mapear combinação de teclas para action_idx)
    action_indices = []
    for _, action in demonstrations:
        # Encontrar qual das 7 ações do RL controller corresponde a essa ação
        for idx, rl_action in enumerate(model.actions):
            if np.array_equal(action, rl_action):
                action_indices.append(idx)
                break
        else:
            # Se não encontrou match exato, use ação neutra (0,0,0,0) = índice 0
            action_indices.append(0)

    actions = torch.tensor(action_indices, dtype=torch.long)

    # Otimizador
    optimizer = torch.optim.Adam(model.policy_net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Treinar
    batch_size = 64
    num_samples = len(states)

    for epoch in range(epochs):
        total_loss = 0
        correct = 0

        # Mini-batches
        indices = torch.randperm(num_samples)
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]

            # Forward pass
            predictions = model.policy_net(batch_states)
            loss = criterion(predictions, batch_actions)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Acurácia
            _, predicted = torch.max(predictions, 1)
            correct += (predicted == batch_actions).sum().item()

        avg_loss = total_loss / (num_samples / batch_size)
        accuracy = 100 * correct / num_samples

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acurácia: {accuracy:.2f}%")

    print(f"\n{'='*60}")
    print(f"TREINAMENTO CONCLUÍDO")
    print(f"Acurácia final: {accuracy:.2f}%")
    print(f"{'='*60}\n")

    return model


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Criar modelo warm-start via imitation learning')
    parser.add_argument('--track', type=str, default='pista1',
                        help=f'Pista para treinar (default: pista1, opções: {list(config.TRACKS.keys())})')
    parser.add_argument('--episodes', type=int, default=20,
                        help='Número de episódios de demonstração (default: 20)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Número de épocas de treinamento (default: 100)')

    args = parser.parse_args()

    # Validar pista
    if args.track not in config.TRACKS:
        print(f"❌ Erro: Pista '{args.track}' não existe.")
        print(f"   Pistas disponíveis: {list(config.TRACKS.keys())}")
        return

    config.set_active_track(args.track)
    track_config = config.get_active_track()

    if track_config['track_string'] is None:
        print(f"❌ Erro: Pista '{args.track}' não está definida ainda.")
        return

    print(f"\n{'='*60}")
    print(f"CRIANDO MODELO WARM-START VIA IMITATION LEARNING")
    print(f"Pista: {track_config['name']}")
    print(f"{'='*60}\n")
    print("Este script vai:")
    print("1. Rodar o A* várias vezes para coletar demonstrações")
    print("2. Treinar uma rede neural para imitar o A*")
    print(f"3. Salvar o modelo como {track_config['warmstart_model']}")
    print(f"\n{'='*60}\n")

    # Coletar demonstrações
    demonstrations = collect_expert_demonstrations(num_episodes=args.episodes)

    # Treinar modelo
    model = train_imitation_model(demonstrations, epochs=args.epochs)

    # Salvar modelo
    save_path = f"{config.MODEL_DIR}/{track_config['warmstart_model']}"
    model.save(save_path)

    print(f"\n{'='*60}")
    print(f"✓ MODELO WARM-START SALVO: {save_path}")
    print(f"{'='*60}\n")
    print("Use este modelo com:")
    print(f"  python scripts/continue_training.py --track {args.track} --generations 100")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
