"""
Treinamento de modelo GENERALISTA - Funciona em múltiplas pistas.

Este script treina um único modelo que aprende a navegar em diferentes pistas.
Usa rotação de pistas durante o treinamento para generalização.

Diferenças do treinamento especializado:
- Treina em múltiplas pistas simultaneamente (curriculum learning)
- Estado expandido com informação da pista
- Mais gerações necessárias para convergência
- Modelo mais robusto mas pode ser menos otimizado por pista

Uso:
    python train_generalist.py --generations 200 --population 50
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pygame
import copy
import random
import argparse
import json
import math
from datetime import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('MacOSX')  # Use native macOS backend
import matplotlib.pyplot as plt

import config
from game.entities.track import Track
from game.entities.kart import Kart
from rl.rl_controller import RLController
from rl.dqn import DQN


class GeneralistTrainer:
    """Treina um modelo generalista em múltiplas pistas."""

    def __init__(self, population_size=50, generations=200, fps=240,
                 track_filter=None, warmstart_path=None, visual_mode=False, plot_live=False):
        self.population_size = population_size
        self.generations = generations
        self.fps = fps
        self.warmstart_path = warmstart_path
        self.visual_mode = visual_mode
        self.plot_live = plot_live

        # Pistas disponíveis para treinamento
        all_tracks = [track_id for track_id, track_data in config.TRACKS.items()
                      if track_data['track_string'] is not None]

        # Filtrar pistas se especificado
        if track_filter:
            requested_tracks = [t.strip() for t in track_filter.split(',')]
            self.available_tracks = [t for t in requested_tracks if t in all_tracks]

            # Validar
            invalid_tracks = [t for t in requested_tracks if t not in all_tracks]
            if invalid_tracks:
                print(f"⚠️  AVISO: Pistas inválidas ignoradas: {invalid_tracks}")

            if not self.available_tracks:
                print(f"❌ ERRO: Nenhuma pista válida especificada!")
                print(f"   Pistas disponíveis: {all_tracks}")
                raise ValueError("Nenhuma pista válida")
        else:
            self.available_tracks = all_tracks

        if len(self.available_tracks) < 2 and not track_filter:
            print("⚠️  AVISO: Apenas 1 pista disponível. Modelo generalista precisa de múltiplas pistas!")
            print("   Continuando com treinamento em pista única...")

        # Métricas
        self.metrics = {
            'best_fitness_per_generation': [],
            'avg_fitness_per_generation': [],
            'completion_rate_per_generation': [],
            'best_fitness_per_track': defaultdict(list),
            'generations': generations,
            'population_size': population_size
        }

        # Hall of Fame global
        self.hall_of_fame = []

        # Visual mode setup
        if self.visual_mode:
            self.screen = None
            self.font = None
            self.clock = pygame.time.Clock()
            self.kart_colors = self._generate_colors(population_size)

        # Plot live setup
        if self.plot_live:
            plt.ion()  # Interactive mode
            self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
            self.fig.suptitle('Treinamento Generalista - Evolução em Tempo Real')

            # Configurar subplots
            self.axes[0, 0].set_title('Best Fitness por Geração')
            self.axes[0, 0].set_xlabel('Geração')
            self.axes[0, 0].set_ylabel('Fitness')
            self.axes[0, 0].grid(True)

            self.axes[0, 1].set_title('Fitness Médio por Geração')
            self.axes[0, 1].set_xlabel('Geração')
            self.axes[0, 1].set_ylabel('Fitness')
            self.axes[0, 1].grid(True)

            self.axes[1, 0].set_title('Taxa de conclusão por Geração')
            self.axes[1, 0].set_xlabel('Geração')
            self.axes[1, 0].set_ylabel('% Completaram')
            self.axes[1, 0].grid(True)

            self.axes[1, 1].set_title('Best Fitness por Pista')
            self.axes[1, 1].set_xlabel('Geração')
            self.axes[1, 1].set_ylabel('Fitness')
            self.axes[1, 1].grid(True)

            plt.tight_layout()
            plt.show(block=False)

    def create_population(self):
        """Cria população inicial de controladores."""
        population = []

        # Se warm start especificado, carregar modelo base
        if self.warmstart_path:
            print(f"\n🔥 Carregando warm start model: {self.warmstart_path}")
            try:
                base_controller = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
                base_controller.load(self.warmstart_path)
                base_controller.epsilon = 0.0
                print(f"   ✓ Modelo carregado com sucesso!")

                # Criar população a partir do modelo base com mutações
                print(f"   📊 Criando população de {self.population_size} variantes...")

                for i in range(self.population_size):
                    controller = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
                    # Copiar pesos do modelo base
                    controller.policy_net.load_state_dict(base_controller.policy_net.state_dict())
                    controller.target_net.load_state_dict(base_controller.target_net.state_dict())
                    controller.epsilon = 0.0

                    # Aplicar mutação (exceto o primeiro, que é cópia exata)
                    if i > 0:
                        mutation_strength = 0.05 + (i / self.population_size) * 0.15  # 5% a 20%
                        with torch.no_grad():
                            for param in controller.policy_net.parameters():
                                if random.random() < 0.3:  # 30% chance de mutar cada camada
                                    noise = torch.randn_like(param) * mutation_strength
                                    param.add_(noise)

                    population.append(controller)

                print(f"   ✓ População criada com variações do warm start!")

            except Exception as e:
                print(f"   ❌ Erro ao carregar warm start: {e}")
                print(f"   Criando população aleatória...")
                self.warmstart_path = None  # Desabilitar warm start

        # Criar população aleatória se não há warm start
        if not self.warmstart_path:
            print(f"\n📊 Criando população aleatória de {self.population_size} controladores...")
            for i in range(self.population_size):
                controller = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
                controller.epsilon = 0.0  # Sem exploração aleatória durante evolução
                population.append(controller)

        return population

    def _generate_colors(self, n):
        """Gera n cores distintas."""
        colors = []
        for i in range(n):
            hue = i * (360 / n)
            # Converter HSV para RGB
            h = hue / 60
            c = 255
            x = int(c * (1 - abs(h % 2 - 1)))

            if h < 1:
                rgb = (c, x, 0)
            elif h < 2:
                rgb = (x, c, 0)
            elif h < 3:
                rgb = (0, c, x)
            elif h < 4:
                rgb = (0, x, c)
            elif h < 5:
                rgb = (x, 0, c)
            else:
                rgb = (c, 0, x)

            colors.append(rgb)

        return colors

    def _update_live_plot(self):
        """Atualiza gráficos em tempo real."""
        if not self.plot_live:
            return

        # Limpar plots
        for ax in self.axes.flatten():
            ax.clear()

        generations = list(range(1, len(self.metrics['best_fitness_per_generation']) + 1))

        # Plot 1: Best Fitness
        self.axes[0, 0].plot(generations, self.metrics['best_fitness_per_generation'], 'g-', linewidth=2)
        self.axes[0, 0].set_title('Best Fitness por Geração')
        self.axes[0, 0].set_xlabel('Geração')
        self.axes[0, 0].set_ylabel('Fitness')
        self.axes[0, 0].grid(True)

        # Plot 2: Avg Fitness
        self.axes[0, 1].plot(generations, self.metrics['avg_fitness_per_generation'], 'b-', linewidth=2)
        self.axes[0, 1].set_title('Fitness Médio por Geração')
        self.axes[0, 1].set_xlabel('Geração')
        self.axes[0, 1].set_ylabel('Fitness')
        self.axes[0, 1].grid(True)

        # Plot 3: Completion Rate
        if self.metrics['completion_rate_per_generation']:
            self.axes[1, 0].plot(generations, self.metrics['completion_rate_per_generation'], 'r-', linewidth=2)
            self.axes[1, 0].set_title('Taxa de conclusão por Geração')
            self.axes[1, 0].set_xlabel('Geração')
            self.axes[1, 0].set_ylabel('% Completaram')
            self.axes[1, 0].set_ylim([0, 105])
            self.axes[1, 0].grid(True)

        # Plot 4: Per-track fitness (com suporte a curriculum learning)
        colors_plot = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, track_id in enumerate(self.available_tracks):
            if track_id in self.metrics['best_fitness_per_track']:
                track_fitness = self.metrics['best_fitness_per_track'][track_id]
                if track_fitness:  # Se tem dados para esta pista
                    # Gerar x-axis apenas para as gerações que têm dados desta pista
                    track_generations = list(range(len(generations) - len(track_fitness) + 1, len(generations) + 1))
                    color = colors_plot[idx % len(colors_plot)]
                    self.axes[1, 1].plot(track_generations,
                                        track_fitness,
                                        color=color, linewidth=2, label=track_id)

        self.axes[1, 1].set_title('Best Fitness por Pista')
        self.axes[1, 1].set_xlabel('Geração')
        self.axes[1, 1].set_ylabel('Fitness')
        self.axes[1, 1].legend()
        self.axes[1, 1].grid(True)

        plt.tight_layout()
        plt.pause(0.001)  # Non-blocking update

    def _evaluate_population_multi_track_visual(self, population, generation_num=0):
        """Avalia população em múltiplas pistas, mostrando cada pista visualmente."""
        print(f"\n  🎮 Modo Visual Multi-Pista: Mostrando {len(self.available_tracks)} pistas sequencialmente")

        # Resultados acumulados de todas as pistas
        accumulated_results = {i: {'fitness': 0, 'track_results': {}} for i in range(len(population))}

        # Avaliar cada pista visualmente
        for track_id in self.available_tracks:
            print(f"\n  📍 Avaliando {len(population)} karts em: {track_id}")

            # Chamar avaliação visual para esta pista específica
            # Temporariamente mudar available_tracks para apenas esta pista
            original_tracks = self.available_tracks
            self.available_tracks = [track_id]

            # Avaliar visualmente nesta pista
            track_results = self._evaluate_single_track_visual(population, track_id, generation_num)

            # Restaurar tracks
            self.available_tracks = original_tracks

            if track_results is None:  # User closed window
                return None

            # Acumular resultados
            for i, result in enumerate(track_results):
                accumulated_results[i]['fitness'] += result['fitness']
                accumulated_results[i]['track_results'][track_id] = result['track_result']

        # Converter para formato esperado e calcular FITNESS MULTIPLICATIVO
        # Usando média geométrica: força modelo a ser BOM em TODAS as pistas
        # (f1 * f2 * f3)^(1/3) - penaliza falhas, recompensa consistência
        final_results = []
        for i in range(len(population)):
            # Coletar fitness de cada pista (normalizado para positivo)
            fitness_values = []
            for track_id in self.available_tracks:
                track_fitness = accumulated_results[i]['track_results'][track_id]['fitness']
                # Normalizar para positivo (somar offset para evitar valores negativos/zero)
                normalized_fitness = max(0, track_fitness + 50000)  # Offset para garantir positivo
                fitness_values.append(normalized_fitness)

            # Média geométrica
            product = 1.0
            for f in fitness_values:
                product *= f
            geometric_mean = product ** (1.0 / len(fitness_values))

            # Remover offset
            final_fitness = geometric_mean - 50000

            final_results.append({
                'controller': population[i],
                'fitness': final_fitness,
                'track_results': accumulated_results[i]['track_results']
            })

        return final_results

    def _evaluate_single_track_visual(self, population, track_id, generation_num):
        """Avalia população em uma única pista com visualização."""
        from game.elements.grass import Grass
        from game.elements.lava import Lava

        track_config = config.TRACKS[track_id]

        # Criar track
        track = Track(
            track_config['track_string'],
            track_config['initial_position'],
            track_config['initial_angle']
        )

        # Atualizar título da janela
        pygame.display.set_caption(f"Gen {generation_num + 1} - {track_id} ({len(population)} karts)")

        # Criar karts
        karts_data = []
        for i, controller in enumerate(population):
            kart = Kart(controller)
            offset_y = (i % 5) * 10
            offset_x = (i // 5) * 10
            initial_pos = [
                track_config['initial_position'][0] + offset_x,
                track_config['initial_position'][1] + offset_y
            ]
            kart.reset(initial_pos, track_config['initial_angle'])
            track.add_kart(kart)

            karts_data.append({
                'controller': controller,
                'kart': kart,
                'id': i,
                'steps': 0,
                'checkpoints': 0,
                'grass_frames': 0,
                'lava_hits': 0,
                'completed': False,
                'alive': True
            })

        # Simular episódio
        max_steps = config.MAX_STEPS_PER_EPISODE
        for step in range(max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    print("\n⚠️  ESC pressionado - Encerrando treinamento...")
                    pygame.quit()
                    sys.exit(0)

            alive_count = sum(1 for kd in karts_data if kd['alive'])
            if alive_count == 0:
                break

            # Atualizar karts vivos
            for kd in karts_data:
                if not kd['alive']:
                    continue

                kart = kd['kart']
                controller = kd['controller']

                keys = controller.move(track_config['track_string'])
                if keys[pygame.K_UP]: kart.forward()
                if keys[pygame.K_DOWN]: kart.backward()
                if keys[pygame.K_LEFT]: kart.turn_left()
                if keys[pygame.K_RIGHT]: kart.turn_right()

                track_element, _ = kart.get_track_element(
                    track_config['track_string'],
                    kart.position[0],
                    kart.position[1]
                )

                if track_element == Grass:
                    kd['grass_frames'] += 1
                if track_element == Lava:
                    kd['lava_hits'] += 1

                kart.update_position(track_config['track_string'], self.screen)
                kd['steps'] += 1

                if kart.has_finished:
                    kd['completed'] = True
                    kd['checkpoints'] = 4
                    kd['alive'] = False
                else:
                    kd['checkpoints'] = kart.next_checkpoint_id

            # Renderizar
            render_freq = 3 if len(population) <= 50 else 5
            if step % render_freq == 0:
                self.screen.fill((0, 0, 0))

                for track_object in track.track_objects:
                    track_object.draw(self.screen)

                for kd in karts_data:
                    if kd['alive']:
                        color = self.kart_colors[kd['id'] % len(self.kart_colors)]
                        self._draw_colored_kart(kd['kart'], color)

                info_texts = [
                    f"Geracao {generation_num + 1}",
                    f"Pista: {track_id}",
                    f"Passo: {step}/{max_steps}",
                    f"Karts vivos: {alive_count}/{len(population)}",
                    f"Completos: {sum(1 for kd in karts_data if kd['completed'])}"
                ]
                y = 10
                for text_str in info_texts:
                    text = self.font.render(text_str, True, (255, 255, 255))
                    self.screen.blit(text, (10, y))
                    y += 25

                pygame.display.flip()

            self.clock.tick(self.fps)

        # Calcular fitness
        results = []
        for kd in karts_data:
            if kd['completed']:
                fitness = config.COMPLETION_BASE_REWARD
                target = track_config.get('target_steps', 300)
                astar_bench = track_config.get('astar_benchmark', 500)

                if kd['steps'] < target:
                    fitness += config.SPEED_EXCELLENCE_BONUS + (target - kd['steps']) * 100
                elif kd['steps'] < astar_bench:
                    fitness += config.BEAT_ASTAR_BONUS + (astar_bench - kd['steps']) * 50

                fitness -= kd['grass_frames'] * config.GRASS_PENALTY_PER_FRAME
                fitness -= kd['lava_hits'] * config.LAVA_PENALTY
            else:
                fitness = kd['checkpoints'] * config.CHECKPOINT_PARTIAL_REWARD
                fitness -= config.TIMEOUT_PENALTY
                fitness -= kd['grass_frames'] * config.GRASS_PENALTY_PER_FRAME
                fitness -= kd['lava_hits'] * config.LAVA_PENALTY

            track_result = {
                'fitness': fitness,
                'completed': kd['completed'],
                'steps': kd['steps'],
                'checkpoints': kd['checkpoints'],
                'grass_frames': kd['grass_frames'],
                'lava_hits': kd['lava_hits']
            }

            results.append({
                'fitness': fitness,
                'track_result': track_result
            })

        return results

    def _evaluate_population_visual(self, population, generation_num=0):
        """Avalia toda a população simultaneamente em modo visual."""
        from game.elements.grass import Grass
        from game.elements.lava import Lava

        # Se múltiplas pistas, avaliar cada uma sequencialmente com visual
        if len(self.available_tracks) > 1:
            return self._evaluate_population_multi_track_visual(population, generation_num)

        # Apenas 1 pista - renderização normal
        track_id = self.available_tracks[0]
        track_config = config.TRACKS[track_id]

        # Criar track
        track = Track(
            track_config['track_string'],
            track_config['initial_position'],
            track_config['initial_angle']
        )

        # Criar karts para todos os indivíduos
        karts_data = []
        for i, controller in enumerate(population):
            kart = Kart(controller)

            # Offset de posição inicial para evitar colisão visual (como train_evolution.py)
            offset_y = (i % 5) * 10
            offset_x = (i // 5) * 10
            initial_pos = [
                track_config['initial_position'][0] + offset_x,
                track_config['initial_position'][1] + offset_y
            ]
            kart.reset(initial_pos, track_config['initial_angle'])

            track.add_kart(kart)

            karts_data.append({
                'controller': controller,
                'kart': kart,
                'id': i,
                'steps': 0,
                'checkpoints': 0,
                'grass_frames': 0,
                'lava_hits': 0,
                'completed': False,
                'alive': True
            })

        # Simular episódio com todos os karts
        max_steps = config.MAX_STEPS_PER_EPISODE
        for step in range(max_steps):
            # Eventos pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

            alive_count = sum(1 for kd in karts_data if kd['alive'])
            if alive_count == 0:
                break

            # Atualizar cada kart vivo
            for kd in karts_data:
                if not kd['alive']:
                    continue

                kart = kd['kart']
                controller = kd['controller']

                # Obter ação
                keys = controller.move(track_config['track_string'])

                if keys[pygame.K_UP]:
                    kart.forward()
                if keys[pygame.K_DOWN]:
                    kart.backward()
                if keys[pygame.K_LEFT]:
                    kart.turn_left()
                if keys[pygame.K_RIGHT]:
                    kart.turn_right()

                # Verificar terreno
                track_element, _ = kart.get_track_element(
                    track_config['track_string'],
                    kart.position[0],
                    kart.position[1]
                )

                if track_element == Grass:
                    kd['grass_frames'] += 1
                if track_element == Lava:
                    kd['lava_hits'] += 1

                # Atualizar posição
                kart.update_position(track_config['track_string'], self.screen)
                kd['steps'] += 1

                # Verificar se completou
                if kart.has_finished:
                    kd['completed'] = True
                    kd['checkpoints'] = 4
                    kd['alive'] = False
                else:
                    kd['checkpoints'] = kart.next_checkpoint_id

            # Renderizar menos frequentemente com muitos karts (a cada 3-5 frames)
            render_freq = 3 if len(population) <= 50 else 5
            if step % render_freq == 0:
                self.screen.fill((0, 0, 0))

                # Desenhar pista
                for track_object in track.track_objects:
                    track_object.draw(self.screen)

                # Desenhar karts vivos
                for kd in karts_data:
                    if kd['alive']:
                        color = self.kart_colors[kd['id'] % len(self.kart_colors)]
                        self._draw_colored_kart(kd['kart'], color)

                # Info na tela
                info_texts = [
                    f"Geracao {generation_num + 1}",
                    f"Pista: {track_id}",
                    f"Passo: {step}/{max_steps}",
                    f"Karts vivos: {alive_count}/{len(population)}",
                    f"Completos: {sum(1 for kd in karts_data if kd['completed'])}"
                ]
                y = 10
                for text_str in info_texts:
                    text = self.font.render(text_str, True, (255, 255, 255))
                    self.screen.blit(text, (10, y))
                    y += 25

                pygame.display.flip()

            self.clock.tick(self.fps)

        # Calcular fitness de cada kart
        results = []
        for kd in karts_data:
            # Calcular fitness
            if kd['completed']:
                fitness = config.COMPLETION_BASE_REWARD

                target = track_config.get('target_steps', 300)
                astar_bench = track_config.get('astar_benchmark', 500)

                if kd['steps'] < target:
                    fitness += config.SPEED_EXCELLENCE_BONUS
                    fitness += (target - kd['steps']) * 100
                elif kd['steps'] < astar_bench:
                    fitness += config.BEAT_ASTAR_BONUS
                    fitness += (astar_bench - kd['steps']) * 50

                fitness -= kd['grass_frames'] * config.GRASS_PENALTY_PER_FRAME
                fitness -= kd['lava_hits'] * config.LAVA_PENALTY
            else:
                fitness = kd['checkpoints'] * config.CHECKPOINT_PARTIAL_REWARD
                fitness -= config.TIMEOUT_PENALTY
                fitness -= kd['grass_frames'] * config.GRASS_PENALTY_PER_FRAME
                fitness -= kd['lava_hits'] * config.LAVA_PENALTY

            track_result = {
                'fitness': fitness,
                'completed': kd['completed'],
                'steps': kd['steps'],
                'checkpoints': kd['checkpoints'],
                'grass_frames': kd['grass_frames'],
                'lava_hits': kd['lava_hits']
            }

            results.append({
                'controller': kd['controller'],
                'fitness': fitness,
                'track_results': {track_id: track_result}
            })

            print(f"  Indivíduo {kd['id']+1}/{len(population)}: Fitness = {fitness:.0f}")

        return results

    def _draw_colored_kart(self, kart, color):
        """Desenha kart com cor específica."""
        kart_position = [int(kart.position[0]), int(kart.position[1])]
        KART_RADIUS = 15

        pygame.draw.circle(self.screen, color, kart_position, KART_RADIUS)

        # Direção
        triangle_size = 15
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

        pygame.draw.polygon(self.screen, (255, 255, 255), vertices)

    def evaluate_on_track(self, controller, track_id, visual_track_obj=None, individual_id=None):
        """Avalia um controlador em uma pista específica."""
        track_config = config.TRACKS[track_id]

        kart = Kart(controller)

        # Se visual mode, usar track passado; senão criar novo
        if visual_track_obj:
            track = visual_track_obj
            screen = self.screen
            clock = self.clock
            use_visual = True
        else:
            track = Track(
                track_config['track_string'],
                track_config['initial_position'],
                track_config['initial_angle']
            )
            pygame.init()
            screen = pygame.display.set_mode((track.width, track.height))
            clock = pygame.time.Clock()
            use_visual = False

        track.add_kart(kart)

        steps = 0
        max_steps = config.MAX_STEPS_PER_EPISODE
        completed = False
        checkpoints_reached = 0
        grass_frames = 0
        lava_hits = 0

        while steps < max_steps and not kart.has_finished:
            # Eventos pygame (para fechar janela)
            if use_visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return None

            # Mover kart
            keys = controller.move(track_config['track_string'])

            if keys[pygame.K_UP]:
                kart.forward()
            if keys[pygame.K_DOWN]:
                kart.backward()
            if keys[pygame.K_LEFT]:
                kart.turn_left()
            if keys[pygame.K_RIGHT]:
                kart.turn_right()

            # Track de grass
            track_element = kart.get_track_element(track_config['track_string'],
                                                   kart.position[0],
                                                   kart.position[1])[0]
            from game.elements.grass import Grass
            from game.elements.lava import Lava

            if track_element == Grass:
                grass_frames += 1

            if track_element == Lava:
                lava_hits += 1

            kart.update_position(track_config['track_string'], screen)

            # Renderizar se visual mode (a cada 3 frames para performance)
            if use_visual and steps % 3 == 0:
                screen.fill((0, 0, 0))

                # Desenhar pista
                for track_object in track.track_objects:
                    track_object.draw(screen)

                # Desenhar kart
                if individual_id is not None and individual_id < len(self.kart_colors):
                    self._draw_colored_kart(kart, self.kart_colors[individual_id])
                else:
                    self._draw_colored_kart(kart, (255, 255, 255))

                # Info no topo
                info_texts = [
                    f"Pista: {track_id}",
                    f"Indivíduo: {individual_id + 1 if individual_id is not None else '?'}",
                    f"Passo: {steps}/{max_steps}",
                    f"CPs: {kart.next_checkpoint_id}/4"
                ]
                y = 10
                for text_str in info_texts:
                    text = self.font.render(text_str, True, (255, 255, 255))
                    screen.blit(text, (10, y))
                    y += 25

                pygame.display.flip()

            clock.tick(self.fps)
            steps += 1

        completed = kart.has_finished
        checkpoints_reached = kart.next_checkpoint_id if not completed else 4

        # Só fazer quit do pygame se não estiver em visual mode
        if not use_visual:
            pygame.quit()

        # Calcular fitness (mesmo sistema que train_evolution.py)
        fitness = 0

        if completed and checkpoints_reached == 4:
            # COMPLETOU! Fitness massivo
            fitness = config.COMPLETION_BASE_REWARD

            # Bônus por velocidade
            target = track_config.get('target_steps', 300)
            astar_bench = track_config.get('astar_benchmark', 500)

            if steps < target:
                fitness += config.SPEED_EXCELLENCE_BONUS
                fitness += (target - steps) * 100
            elif steps < astar_bench:
                fitness += config.BEAT_ASTAR_BONUS
                fitness += (astar_bench - steps) * 50

            # Penalidades por comportamento ruim
            fitness -= grass_frames * config.GRASS_PENALTY_PER_FRAME
            fitness -= lava_hits * config.LAVA_PENALTY
        else:
            # NÃO COMPLETOU - fitness muito baixo
            fitness = checkpoints_reached * config.CHECKPOINT_PARTIAL_REWARD
            fitness -= config.TIMEOUT_PENALTY
            fitness -= grass_frames * config.GRASS_PENALTY_PER_FRAME
            fitness -= lava_hits * config.LAVA_PENALTY

        return {
            'fitness': fitness,
            'completed': completed,
            'steps': steps,
            'checkpoints': checkpoints_reached,
            'grass_frames': grass_frames,
            'lava_hits': lava_hits
        }

    def evaluate_individual(self, controller, individual_id=None):
        """Avalia um indivíduo em TODAS as pistas disponíveis."""
        total_fitness = 0
        track_results = {}

        for track_id in self.available_tracks:
            # Se visual mode, criar track object para passar
            if self.visual_mode:
                track_config = config.TRACKS[track_id]
                visual_track = Track(
                    track_config['track_string'],
                    track_config['initial_position'],
                    track_config['initial_angle']
                )
                result = self.evaluate_on_track(controller, track_id, visual_track, individual_id)
            else:
                result = self.evaluate_on_track(controller, track_id)

            if result is None:  # User closed window
                return None

            total_fitness += result['fitness']
            track_results[track_id] = result

        # Fitness é a média entre todas as pistas
        avg_fitness = total_fitness / len(self.available_tracks)

        return {
            'fitness': avg_fitness,
            'track_results': track_results
        }

    def run_generation(self, population, generation_num):
        """Executa uma geração completa."""
        print(f"\n{'='*70}")
        print(f"GERAÇÃO {generation_num + 1}/{self.generations}")
        print(f"{'='*70}")

        # CURRICULUM LEARNING: Adicionar pistas gradualmente
        # Gen 1-100: Só pista1 (dominar a básica)
        # Gen 101-200: pista1 + pista_retangular (adicionar complexidade)
        # Gen 201+: Todas as 3 pistas (generalista completo)
        original_tracks = self.available_tracks.copy()

        if generation_num < 100:
            # Fase 1: Dominar pista básica
            self.available_tracks = [t for t in original_tracks if t == 'pista1']
            print(f"📚 CURRICULUM Fase 1: Dominando pista básica ({len(self.available_tracks)} pista)")
        elif generation_num < 200:
            # Fase 2: Adicionar pista retangular
            self.available_tracks = [t for t in original_tracks if t in ['pista1', 'pista_retangular']]
            print(f"📚 CURRICULUM Fase 2: Adicionando complexidade ({len(self.available_tracks)} pistas)")
        else:
            # Fase 3: Todas as pistas
            self.available_tracks = original_tracks
            print(f"📚 CURRICULUM Fase 3: Generalista completo ({len(self.available_tracks)} pistas)")

        # Se visual mode, inicializar pygame/screen uma vez
        if self.visual_mode:
            if self.screen is None:
                pygame.init()
                # Usar primeira pista para determinar tamanho da janela
                first_track_config = config.TRACKS[self.available_tracks[0]]
                sample_track = Track(
                    first_track_config['track_string'],
                    first_track_config['initial_position'],
                    first_track_config['initial_angle']
                )
                self.screen = pygame.display.set_mode((sample_track.width, sample_track.height))
                pygame.display.set_caption(f"Generalist Training - Gen {generation_num + 1}")
                self.font = pygame.font.Font(None, 24)

        # Avaliar todos os indivíduos
        if self.visual_mode:
            # Modo visual: avaliar todos simultaneamente
            results = self._evaluate_population_visual(population, generation_num)
            if results is None:
                print("\n⚠️  Treinamento interrompido pelo usuário")
                return None
        else:
            # Modo não-visual: avaliar sequencialmente (mais rápido)
            results = []
            for i, controller in enumerate(population):
                result = self.evaluate_individual(controller, individual_id=i)

                if result is None:
                    print("\n⚠️  Treinamento interrompido pelo usuário")
                    return None

                results.append({
                    'controller': controller,
                    'fitness': result['fitness'],
                    'track_results': result['track_results']
                })

                print(f"  Indivíduo {i+1}/{self.population_size}: Fitness = {result['fitness']:.0f}")

        # Ordenar por fitness
        results.sort(key=lambda x: x['fitness'], reverse=True)

        # Estatísticas
        best_fitness = results[0]['fitness']
        avg_fitness = sum(r['fitness'] for r in results) / len(results)

        # Taxa de sucesso por pista e overall
        total_completions = 0
        total_attempts = 0
        for track_id in self.available_tracks:
            completions = sum(1 for r in results if r['track_results'][track_id]['completed'])
            rate = (completions / len(results)) * 100

            # Encontrar melhor resultado (menor steps entre os que completaram, ou melhor fitness)
            completed_on_track = [r for r in results if r['track_results'][track_id]['completed']]
            if completed_on_track:
                best_on_track = min(completed_on_track, key=lambda r: r['track_results'][track_id]['steps'])
                best_steps = best_on_track['track_results'][track_id]['steps']
                astar_bench = config.TRACKS[track_id]['astar_benchmark']
                efficiency = (astar_bench / best_steps * 100) if best_steps > 0 else 0
                print(f"\n  {track_id}: {completions}/{len(results)} completaram ({rate:.1f}%) | Melhor: {best_steps} passos ({efficiency:.1f}% vs A*)")
            else:
                print(f"\n  {track_id}: {completions}/{len(results)} completaram ({rate:.1f}%) | Nenhum completou")

            total_completions += completions
            total_attempts += len(results)

        overall_completion_rate = (total_completions / total_attempts) * 100 if total_attempts > 0 else 0

        print(f"\n  Melhor Fitness: {best_fitness:.0f}")
        print(f"  Fitness Médio: {avg_fitness:.0f}")
        print(f"  Taxa de conclusão Overall: {overall_completion_rate:.1f}%")

        # Salvar métricas
        self.metrics['best_fitness_per_generation'].append(best_fitness)
        self.metrics['avg_fitness_per_generation'].append(avg_fitness)
        self.metrics['completion_rate_per_generation'].append(overall_completion_rate)

        for track_id in self.available_tracks:
            best_on_track = max(r['track_results'][track_id]['fitness'] for r in results)
            self.metrics['best_fitness_per_track'][track_id].append(best_on_track)

        # Atualizar Hall of Fame
        for r in results[:config.HALL_OF_FAME_SIZE]:
            self.hall_of_fame.append({
                'generation': generation_num,
                'fitness': r['fitness'],
                'controller': copy.deepcopy(r['controller'])
            })

        # Manter apenas os melhores no Hall of Fame
        self.hall_of_fame.sort(key=lambda x: x['fitness'], reverse=True)
        self.hall_of_fame = self.hall_of_fame[:config.HALL_OF_FAME_SIZE]

        # Nova geração
        new_population = []

        # Elite passa direto
        for i in range(config.ELITE_SIZE):
            new_population.append(copy.deepcopy(results[i]['controller']))

        # Crossover + mutação
        mutation_rate = max(
            config.MUTATION_RATE_MIN,
            config.MUTATION_RATE_INITIAL * (0.95 ** generation_num)
        )

        # SELEÇÃO TRUNCADA: Apenas top 30% podem ser pais
        # Isso acelera convergência e evita que karts presos contaminem a população
        selection_cutoff = max(10, int(len(results) * 0.3))  # Mínimo 10, ou top 30%
        breeding_pool = results[:selection_cutoff]  # Results já está ordenado por fitness

        print(f"  🎯 Seleção truncada: {selection_cutoff}/{len(results)} karts podem reproduzir (top {100*selection_cutoff/len(results):.0f}%)")

        while len(new_population) < self.population_size:
            # Selecionar pais apenas do breeding pool (top performers)
            parent1 = random.choice(breeding_pool)
            parent2 = random.choice(breeding_pool)

            # Crossover
            child = self.crossover(parent1['controller'], parent2['controller'])

            # Mutação
            child = self.mutate(child, mutation_rate)

            new_population.append(child)

        # Atualizar gráficos em tempo real
        self._update_live_plot()

        # Restaurar pistas originais
        self.available_tracks = original_tracks

        return new_population

    def tournament_selection(self, results, k=3):
        """Seleção por torneio (não usado mais - substituído por seleção truncada)."""
        tournament = random.sample(results, k)
        return max(tournament, key=lambda x: x['fitness'])

    def crossover(self, parent1, parent2):
        """Crossover entre dois pais."""
        child = RLController(state_size=config.STATE_SIZE, action_size=config.ACTION_SIZE)
        child.epsilon = 0.0

        # Crossover dos pesos
        for (name1, param1), (name2, param2), (name_child, param_child) in zip(
            parent1.policy_net.named_parameters(),
            parent2.policy_net.named_parameters(),
            child.policy_net.named_parameters()
        ):
            # Escolher aleatoriamente de qual pai herdar cada peso
            mask = torch.rand_like(param1.data) > 0.5
            param_child.data = torch.where(mask, param1.data, param2.data)

        return child

    def mutate(self, controller, mutation_rate):
        """Aplica mutação aos pesos do modelo."""
        for param in controller.policy_net.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(param.data) * 0.1
                param.data += noise
        return controller

    def train(self):
        """Loop principal de treinamento."""
        print("\n" + "="*70)
        print("TREINAMENTO DE MODELO GENERALISTA")
        print("="*70)
        print(f"Pistas: {', '.join(self.available_tracks)}")
        print(f"População: {self.population_size}")
        print(f"Gerações: {self.generations}")
        print("="*70)

        # Criar população inicial
        population = self.create_population()

        # Loop de gerações
        for generation in range(self.generations):
            population = self.run_generation(population, generation)

            # Se None, usuário fechou a janela
            if population is None:
                print("\n" + "="*70)
                print("TREINAMENTO INTERROMPIDO")
                print("="*70)
                return

            # Salvar checkpoint a cada 10 gerações
            if (generation + 1) % 10 == 0:
                best_controller = self.hall_of_fame[0]['controller']
                save_path = f"{config.MODEL_DIR}/generalist_gen{generation + 1}.pth"
                best_controller.save(save_path)
                print(f"\n  💾 Checkpoint salvo: {save_path}")

                # Salvar gráfico intermediário
                if self.plot_live:
                    track_names = '_'.join(self.available_tracks)
                    plot_checkpoint = f"{config.METRICS_DIR}/generalist_{track_names}_gen{generation + 1}.png"
                    self.fig.savefig(plot_checkpoint, dpi=150, bbox_inches='tight')
                    print(f"  📊 Gráfico salvo: {plot_checkpoint}")

        # Salvar modelo final
        best_controller = self.hall_of_fame[0]['controller']
        best_controller.save(f"{config.MODEL_DIR}/{config.GENERALIST_MODEL}")

        # Salvar métricas
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        track_names = '_'.join(self.available_tracks)

        metrics_file = f"{config.METRICS_DIR}/generalist_{track_names}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            # Converter defaultdict para dict normal
            metrics_copy = self.metrics.copy()
            metrics_copy['best_fitness_per_track'] = dict(metrics_copy['best_fitness_per_track'])
            metrics_copy['tracks'] = self.available_tracks
            metrics_copy['timestamp'] = timestamp
            json.dump(metrics_copy, f, indent=2)

        # Salvar gráfico final como PNG
        if self.plot_live:
            plot_file = f"{config.METRICS_DIR}/generalist_{track_names}_plot_{timestamp}.png"
            self._update_live_plot()  # Atualizar uma última vez
            self.fig.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"Gráfico salvo: {plot_file}")

        print("\n" + "="*70)
        print("TREINAMENTO CONCLUÍDO!")
        print("="*70)
        print(f"Melhor Fitness: {self.hall_of_fame[0]['fitness']:.0f}")
        print(f"Modelo salvo: {config.MODEL_DIR}/{config.GENERALIST_MODEL}")
        print(f"Métricas salvas: {metrics_file}")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Treinar modelo generalista')
    parser.add_argument('--generations', type=int, default=200,
                        help='Número de gerações (default: 200)')
    parser.add_argument('--population', type=int, default=50,
                        help='Tamanho da população (default: 50)')
    parser.add_argument('--fps', type=int, default=240,
                        help='FPS da visualização (default: 240)')
    parser.add_argument('--tracks', type=str, default=None,
                        help='Pistas para treinar (separadas por vírgula, ex: pista1,pista_retangular). Default: todas')
    parser.add_argument('--warmstart', type=str, default=None,
                        help='Caminho para modelo warm start (ex: rl/models/pista1_warmstart.pth)')
    parser.add_argument('--visual', action='store_true',
                        help='Habilitar modo visual (renderizar karts correndo)')
    parser.add_argument('--plot-live', action='store_true',
                        help='Plotar gráficos de evolução em tempo real')

    args = parser.parse_args()

    trainer = GeneralistTrainer(
        population_size=args.population,
        generations=args.generations,
        fps=args.fps,
        track_filter=args.tracks,
        warmstart_path=args.warmstart,
        visual_mode=args.visual,
        plot_live=args.plot_live
    )

    trainer.train()


if __name__ == '__main__':
    main()
