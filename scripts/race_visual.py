"""
Modo de corrida visual com múltiplos karts competindo.
Mostra AI A*, RL treinado, e RL em exploração na mesma pista.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pygame

import config
from game.entities.track import Track
from game.entities.kart import Kart
from game.controllers.ai import AI
from rl.rl_controller import RLController


def main():
    """Corrida com múltiplos karts visíveis."""
    import argparse

    parser = argparse.ArgumentParser(description='Corrida visual: A* vs Neural Network')
    parser.add_argument('--track', type=str, default='pista1',
                        help=f'Pista para correr (default: pista1, opções: {list(config.TRACKS.keys())})')
    args = parser.parse_args()

    # Validar e configurar pista
    if args.track not in config.TRACKS:
        print(f"❌ Erro: Pista '{args.track}' não existe.")
        print(f"   Pistas disponíveis: {list(config.TRACKS.keys())}")
        return

    config.set_active_track(args.track)
    track_config = config.get_active_track()

    if track_config['track_string'] is None:
        print(f"❌ Erro: Pista '{args.track}' não está definida ainda.")
        return

    # Criar controladores diferentes
    ai_controller = AI()

    # RL treinado (carregar modelo evolutivo da pista)
    rl_trained = RLController(state_size=12, action_size=7)
    best_model = f"{config.MODEL_DIR}/{track_config['best_model']}"
    warmstart_model = f"{config.MODEL_DIR}/{track_config['warmstart_model']}"

    try:
        rl_trained.load(best_model)
        rl_trained.epsilon = 0.0  # Sem exploração (usa o que aprendeu)
        print(f"Modelo {track_config['best_model']} carregado!")
    except:
        print(f"Modelo best não encontrado, tentando warmstart...")
        try:
            rl_trained.load(warmstart_model)
            rl_trained.epsilon = 0.0
            print(f"Modelo {track_config['warmstart_model']} carregado!")
        except:
            print("Nenhum modelo encontrado, usando não treinado")

    # Criar karts (apenas A* e RL)
    kart_ai = Kart(ai_controller)
    kart_rl_trained = Kart(rl_trained)

    # Criar track
    track = Track(track_config['track_string'], track_config['initial_position'], track_config['initial_angle'])
    track.add_kart(kart_ai)
    track.add_kart(kart_rl_trained)

    # Inicializar Pygame
    pygame.init()
    screen = pygame.display.set_mode((track.width, track.height))
    pygame.display.set_caption(f"Mario Kart - A* vs Neural Network - {track_config['name']}")
    clock = pygame.time.Clock()

    # Resetar karts em posições ligeiramente diferentes
    init_pos = track_config['initial_position']
    init_angle = track_config['initial_angle']
    kart_ai.reset([init_pos[0], init_pos[1]], init_angle)
    kart_rl_trained.reset([init_pos[0], init_pos[1] + 25], init_angle)

    # Cores para cada kart
    kart_colors = [
        (0, 255, 0),    # Verde - A*
        (0, 100, 255)   # Azul - RL Treinado
    ]

    running = True
    cycles = 0
    max_cycles = 2000

    print("\n" + "="*60)
    print(f"CORRIDA: A* vs NEURAL NETWORK - {track_config['name']}")
    print("="*60)
    print("Verde: A* Pathfinding (Otimizado)")
    print("Azul: Rede Neural Evolutiva (Treinada)")
    print("="*60 + "\n")

    while running and cycles < max_cycles:
        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Limpar tela
        screen.fill((0, 0, 0))

        # Desenhar pista
        for track_object in track.track_objects:
            track_object.draw(screen)

        # Atualizar cada kart
        for idx, kart in enumerate(track.karts):
            if not kart.has_finished:
                # Obter comando do controlador
                keys = kart.controller.move(config.DEFAULT_TRACK)

                # Aplicar comandos
                if keys[pygame.K_UP]:
                    kart.forward()
                if keys[pygame.K_DOWN]:
                    kart.backward()
                if keys[pygame.K_LEFT]:
                    kart.turn_left()
                if keys[pygame.K_RIGHT]:
                    kart.turn_right()

                # Atualizar posição
                kart.update_position(config.DEFAULT_TRACK, screen)

                # Desenhar kart com cor específica
                draw_colored_kart(screen, kart, kart_colors[idx])

        # Informações na tela
        font = pygame.font.Font(None, 24)

        # Status de cada kart
        y_offset = 10
        labels = ["A* (Verde)", "Neural Net (Azul)"]

        for idx, (kart, label, color) in enumerate(zip(track.karts, labels, kart_colors)):
            # Checkpoints: 4 total quando has_finished=True, senão next_checkpoint_id
            checkpoints_count = 4 if kart.has_finished else kart.next_checkpoint_id
            status = f"{label}: CP {checkpoints_count}/4"
            if kart.has_finished:
                status += " - FINALIZADO!"

            text = font.render(status, True, color)
            screen.blit(text, (10, y_offset))
            y_offset += 30

        # Ciclos
        text = font.render(f"Ciclos: {cycles}/{max_cycles}", True, (255, 255, 255))
        screen.blit(text, (10, track.height - 30))

        pygame.display.flip()
        clock.tick(60)  # 60 FPS

        cycles += 1

        # Verificar se todos terminaram
        if all(k.has_finished for k in track.karts):
            running = False

    print("\n" + "="*60)
    print("CORRIDA FINALIZADA!")
    print("="*60)

    for idx, (kart, label) in enumerate(zip(track.karts, labels)):
        checkpoints_count = 4 if kart.has_finished else kart.next_checkpoint_id
        if kart.has_finished:
            print(f"{label}: COMPLETOU A CORRIDA! (Checkpoints: 4/4)")
        else:
            print(f"{label}: Não completou (Checkpoints: {checkpoints_count}/4)")

    print(f"\nTotal de ciclos: {cycles}")
    print("="*60)

    # Aguardar alguns segundos antes de fechar
    pygame.time.wait(5000)
    pygame.quit()


def draw_colored_kart(screen, kart, color):
    """Desenha o kart com uma cor específica."""
    import math

    kart_position = [kart.position[0], kart.position[1]]
    KART_RADIUS = 20

    # Círculo do kart
    pygame.draw.circle(screen, color, kart_position, KART_RADIUS)

    # Triângulo indicando direção
    triangle_size = 20
    angle_offset = math.pi / 2

    vertices = [
        (
            kart_position[0] + int(triangle_size * math.cos(kart.angle + angle_offset)),
            kart_position[1] + int(triangle_size * math.sin(kart.angle + angle_offset))
        ),
        (
            kart_position[0] + int(triangle_size * math.cos(kart.angle - angle_offset)),
            kart_position[1] + int(triangle_size * math.sin(kart.angle - angle_offset))
        ),
        (
            kart_position[0] + int(triangle_size * 1.5 * math.cos(kart.angle)),
            kart_position[1] + int(triangle_size * 1.5 * math.sin(kart.angle))
        )
    ]

    pygame.draw.polygon(screen, (255, 255, 255), vertices)


if __name__ == '__main__':
    main()
