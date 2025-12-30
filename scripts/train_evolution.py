"""
Treinamento Evolutivo - Algoritmo Genético para Karts

Múltiplos karts competem simultaneamente.
Os melhores passam para a próxima geração.
Recompensa por velocidade: chegar rápido nos checkpoints!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pygame
import copy
import random
from collections import defaultdict

import config
from game.entities.track import Track
from game.entities.kart import Kart
from rl.rl_controller import RLController
from rl.dqn import DQN


class EvolutionaryKart:
    """Um kart individual na população."""

    def __init__(self, controller, kart_id):
        self.controller = controller
        self.kart_id = kart_id
        self.kart = None

        # Métricas de fitness
        self.fitness = 0
        self.checkpoints_reached = 0
        self.steps_taken = 0
        self.checkpoint_times = []  # Tempo para alcançar cada checkpoint
        self.total_reward = 0
        self.is_alive = True

        # Métricas avançadas
        self.lava_hits = 0
        self.min_distance_to_checkpoint = float('inf')
        self.distance_traveled = 0
        self.last_position = None
        self.stuck_counter = 0  # Contador de frames parado
        self.grass_frames = 0  # Frames na grama
        self.completed = False  # Flag se completou a pista

    def reset(self, position, angle):
        """Reseta o kart para nova corrida."""
        self.kart = Kart(self.controller)
        self.controller.kart = self.kart
        self.kart.reset(position, angle)

        self.fitness = 0
        self.checkpoints_reached = 0
        self.steps_taken = 0
        self.checkpoint_times = []
        self.total_reward = 0
        self.is_alive = True
        self.lava_hits = 0
        self.min_distance_to_checkpoint = float('inf')
        self.distance_traveled = 0
        self.last_position = list(position)
        self.stuck_counter = 0
        self.grass_frames = 0
        self.completed = False

    def calculate_fitness(self):
        """
        NOVA ESTRATÉGIA DE FITNESS - FOCADA EM COMPLETAR + VELOCIDADE

        Filosofia: Se não completa a pista, fitness é RUIM.
        Se completa, fitness é ÓTIMO + bônus por velocidade.

        Objetivo: Bater A* (440 passos) e idealmente < 400 passos
        """

        # CASO 1: COMPLETOU A PISTA (fitness MASSIVO)
        if self.completed and self.checkpoints_reached == 4:
            # Base reward gigante
            self.fitness = config.COMPLETION_BASE_REWARD

            # Bônus por VELOCIDADE (quanto mais rápido, melhor)
            total_steps = self.steps_taken

            # EXCELÊNCIA: Completou em < 400 passos
            if total_steps < config.TARGET_STEPS:
                self.fitness += config.SPEED_EXCELLENCE_BONUS
                # Bônus extra por cada passo economizado
                self.fitness += (config.TARGET_STEPS - total_steps) * 100

            # BOM: Bateu o A* (< 440 passos)
            elif total_steps < config.ASTAR_BENCHMARK_STEPS:
                self.fitness += config.BEAT_ASTAR_BONUS
                # Bônus por cada passo economizado vs A*
                self.fitness += (config.ASTAR_BENCHMARK_STEPS - total_steps) * 50

            # RAZOÁVEL: Completou mas mais devagar que A*
            else:
                # Penalidade por ser mais lento que A*
                slowness_penalty = (total_steps - config.ASTAR_BENCHMARK_STEPS) * 20
                self.fitness -= slowness_penalty

            # Penalidades por comportamento ruim (mesmo completando)
            self.fitness -= self.grass_frames * config.GRASS_PENALTY_PER_FRAME
            self.fitness -= self.lava_hits * config.LAVA_PENALTY
            self.fitness -= self.stuck_counter * config.STUCK_PENALTY

        # CASO 2: NÃO COMPLETOU (fitness muito menor)
        else:
            # Fitness base negativo
            self.fitness = -config.TIMEOUT_PENALTY

            # Pequeno crédito por checkpoints parciais
            self.fitness += self.checkpoints_reached * config.CHECKPOINT_PARTIAL_REWARD

            # Pequeno crédito por progresso ao próximo checkpoint
            if self.min_distance_to_checkpoint < float('inf'):
                progress = max(0, config.PROGRESS_PARTIAL_REWARD * (1 - self.min_distance_to_checkpoint / 1000))
                self.fitness += progress

            # Penalidades ainda aplicam
            self.fitness -= self.grass_frames * config.GRASS_PENALTY_PER_FRAME
            self.fitness -= self.lava_hits * config.LAVA_PENALTY
            self.fitness -= self.stuck_counter * config.STUCK_PENALTY

        return self.fitness


class EvolutionaryTrainer:
    """Treinador evolutivo com visualização."""

    def __init__(self, track_string, initial_position, initial_angle, population_size=10):
        self.track_string = track_string
        self.initial_position = initial_position
        self.initial_angle = initial_angle
        self.population_size = population_size

        # População
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

        # Parâmetros
        self.max_steps_per_episode = config.MAX_STEPS_PER_EPISODE
        self.mutation_rate = config.MUTATION_RATE_INITIAL
        self.elite_size = config.ELITE_SIZE

        # Pygame
        pygame.init()
        self.screen = None
        self.font = None
        self.clock = pygame.time.Clock()
        self.fps = 60  # Default FPS (pode ser alterado externamente)

        # Cores únicas para cada kart
        self.kart_colors = self._generate_colors(population_size)

        # Inicializar população
        self._initialize_population()

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

    def _initialize_population(self):
        """Cria população inicial com redes neurais aleatórias."""
        print(f"Inicializando população de {self.population_size} karts...")

        for i in range(self.population_size):
            controller = RLController(state_size=12, action_size=7, learning_rate=0.001)
            controller.epsilon = 0.0  # Sem exploração randômica, usa evolução

            evo_kart = EvolutionaryKart(controller, i)
            self.population.append(evo_kart)

        print("População inicializada!")

    def _mutate_network(self, network, mutation_rate):
        """Aplica mutação nos pesos da rede neural."""
        with torch.no_grad():
            for param in network.parameters():
                if random.random() < mutation_rate:
                    # Mutação: adicionar ruído gaussiano
                    noise = torch.randn_like(param) * 0.1
                    param.add_(noise)

    def _crossover_networks(self, parent1_net, parent2_net):
        """Combina duas redes neurais (crossover)."""
        child_net = DQN(12, 7).to(parent1_net.fc1.weight.device)

        with torch.no_grad():
            for child_param, parent1_param, parent2_param in zip(
                child_net.parameters(),
                parent1_net.parameters(),
                parent2_net.parameters()
            ):
                # Crossover: média dos pais ou escolha aleatória
                if random.random() < 0.5:
                    child_param.data.copy_(parent1_param.data)
                else:
                    child_param.data.copy_(parent2_param.data)

        return child_net

    def _create_next_generation(self):
        """Cria próxima geração através de seleção, crossover e mutação."""
        # Ordenar por fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Estatísticas
        best_fitness = self.population[0].fitness
        avg_fitness = np.mean([k.fitness for k in self.population])
        worst_fitness = self.population[-1].fitness
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        # Estatísticas detalhadas do melhor
        best = self.population[0]

        print(f"\n{'='*60}")
        print(f"GERAÇÃO {self.generation} COMPLETA")
        print(f"{'='*60}")
        print(f"Melhor kart: #{best.kart_id}")
        print(f"  Fitness: {best.fitness:.1f}")

        if best.completed:
            print(f"  ✓ COMPLETOU a pista!")
            print(f"  Passos: {best.steps_taken} (A* benchmark: {config.ASTAR_BENCHMARK_STEPS})")
            if best.steps_taken < config.TARGET_STEPS:
                print(f"  ⭐ EXCELÊNCIA! Bateu meta de {config.TARGET_STEPS} passos!")
            elif best.steps_taken < config.ASTAR_BENCHMARK_STEPS:
                print(f"  🏆 Bateu A*! ({config.ASTAR_BENCHMARK_STEPS - best.steps_taken} passos mais rápido)")
        else:
            print(f"  ✗ Não completou - Checkpoints: {best.checkpoints_reached}/4")
            print(f"  Passos: {best.steps_taken}")
            print(f"  Dist. min ao CP: {best.min_distance_to_checkpoint:.1f}")

        print(f"  Grama (frames): {best.grass_frames}")
        print(f"  Lava hits: {best.lava_hits}")
        print(f"  Preso (frames): {best.stuck_counter}")
        print(f"\nFitness médio: {avg_fitness:.1f}")
        print(f"Fitness pior: {worst_fitness:.1f}")
        print(f"{'='*60}\n")

        # Mutação adaptativa (diminui com o tempo)
        adaptive_mutation = max(0.05, self.mutation_rate * (0.95 ** self.generation))

        # Nova população
        new_population = []

        # Elite: top 3 passam direto (sem mutação)
        print(f"Elite (passam direto):")
        for i in range(self.elite_size):
            elite = self.population[i]
            new_controller = RLController(state_size=12, action_size=7)
            new_controller.policy_net.load_state_dict(elite.controller.policy_net.state_dict())
            new_controller.epsilon = 0.0
            new_population.append(EvolutionaryKart(new_controller, i))
            print(f"  #{elite.kart_id}: Fitness={elite.fitness:.1f}, CPs={elite.checkpoints_reached}")

        # Resto: crossover + mutação
        while len(new_population) < self.population_size:
            # Seleção por torneio (favorece os melhores)
            tournament_size = 4
            # Pegar da metade superior
            candidates = self.population[:max(5, self.population_size//2)]
            parents = random.sample(candidates, min(tournament_size, len(candidates)))
            parents.sort(key=lambda x: x.fitness, reverse=True)
            parent1, parent2 = parents[0], parents[1] if len(parents) > 1 else parents[0]

            # Criar filho através de crossover
            new_controller = RLController(state_size=12, action_size=7)
            child_net = self._crossover_networks(
                parent1.controller.policy_net,
                parent2.controller.policy_net
            )
            new_controller.policy_net = child_net
            new_controller.target_net.load_state_dict(child_net.state_dict())

            # Mutação adaptativa
            self._mutate_network(new_controller.policy_net, adaptive_mutation)

            new_controller.epsilon = 0.0
            new_population.append(EvolutionaryKart(new_controller, len(new_population)))

        print(f"Taxa de mutação adaptativa: {adaptive_mutation:.3f}")

        self.population = new_population
        self.generation += 1

        # Salvar melhor modelo
        if self.generation % 10 == 0:
            best.controller.save(f"rl/models/evolution_gen{self.generation}.pth")

    def run_generation(self):
        """Executa uma geração completa com visualização."""
        # Criar track
        track = Track(self.track_string, self.initial_position, self.initial_angle)

        if self.screen is None:
            self.screen = pygame.display.set_mode((track.width, track.height))
            pygame.display.set_caption(f"Evolução - Geração {self.generation}")
            self.font = pygame.font.Font(None, 18)

        # Resetar todos os karts
        for evo_kart in self.population:
            # Posições iniciais variadas para evitar colisão visual
            offset_y = (evo_kart.kart_id % 5) * 10
            evo_kart.reset([self.initial_position[0], self.initial_position[1] + offset_y], 0)

        # Executar episódio
        for step in range(self.max_steps_per_episode):
            # Eventos
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False

            # Atualizar cada kart vivo
            alive_count = 0
            for evo_kart in self.population:
                if not evo_kart.is_alive:
                    continue

                alive_count += 1
                kart = evo_kart.kart

                # Obter ação
                state = evo_kart.controller.get_state(self.track_string)
                action_idx = evo_kart.controller.select_action(state, training=False)
                action = evo_kart.controller.actions[action_idx]

                # Aplicar ação
                keys = {
                    pygame.K_UP: action[0],
                    pygame.K_DOWN: action[1],
                    pygame.K_LEFT: action[2],
                    pygame.K_RIGHT: action[3]
                }

                if keys[pygame.K_UP]:
                    kart.forward()
                if keys[pygame.K_DOWN]:
                    kart.backward()
                if keys[pygame.K_LEFT]:
                    kart.turn_left()
                if keys[pygame.K_RIGHT]:
                    kart.turn_right()

                # Salvar estado anterior
                old_checkpoint = kart.next_checkpoint_id
                old_position = list(kart.position)

                # Verificar terreno atual ANTES de mover
                from game.elements.lava import Lava
                from game.elements.grass import Grass
                current_terrain, _ = kart.get_track_element(
                    self.track_string,
                    kart.position[0],
                    kart.position[1]
                )

                # Atualizar posição
                kart.update_position(self.track_string, self.screen)

                # Atualizar métricas básicas
                evo_kart.steps_taken += 1

                # Rastrear distância percorrida
                if evo_kart.last_position is not None:
                    distance = np.sqrt(
                        (kart.position[0] - evo_kart.last_position[0])**2 +
                        (kart.position[1] - evo_kart.last_position[1])**2
                    )
                    evo_kart.distance_traveled += distance

                    # Detectar se ficou preso (moveu menos de 2 pixels)
                    if distance < 2:
                        evo_kart.stuck_counter += 1

                evo_kart.last_position = list(kart.position)

                # Detectar se bateu em lava (resetou posição drasticamente)
                if current_terrain == Lava or (
                    np.sqrt(
                        (kart.position[0] - old_position[0])**2 +
                        (kart.position[1] - old_position[1])**2
                    ) > 100  # Teleportou = lava
                ):
                    evo_kart.lava_hits += 1

                # Detectar se está na grama (penalidade PESADA)
                if current_terrain == Grass:
                    evo_kart.grass_frames += 1

                # Calcular distância ao próximo checkpoint
                checkpoint_chars = ['C', 'D', 'E', 'F']
                if kart.next_checkpoint_id < len(checkpoint_chars):
                    next_cp_char = checkpoint_chars[kart.next_checkpoint_id]
                    rows = self.track_string.split('\n')
                    rows = [row.strip() for row in rows if row.strip()]

                    for row_idx in range(len(rows)):
                        row = rows[row_idx]
                        for col_idx in range(len(row)):
                            if row[col_idx] == next_cp_char:
                                cp_x = col_idx * config.BLOCK_SIZE + config.BLOCK_SIZE // 2
                                cp_y = row_idx * config.BLOCK_SIZE + config.BLOCK_SIZE // 2

                                distance_to_cp = np.sqrt(
                                    (kart.position[0] - cp_x)**2 +
                                    (kart.position[1] - cp_y)**2
                                )

                                # Atualizar distância mínima
                                if distance_to_cp < evo_kart.min_distance_to_checkpoint:
                                    evo_kart.min_distance_to_checkpoint = distance_to_cp

                # Verificar progresso de checkpoint
                if kart.next_checkpoint_id > old_checkpoint:
                    evo_kart.checkpoints_reached = kart.next_checkpoint_id
                    evo_kart.checkpoint_times.append(step)

                    # Resetar distância mínima para próximo checkpoint
                    evo_kart.min_distance_to_checkpoint = float('inf')

                    # Bonus por velocidade!
                    speed_bonus = 100 / max(1, step - (evo_kart.checkpoint_times[-2] if len(evo_kart.checkpoint_times) > 1 else 0))
                    evo_kart.total_reward += 100 + speed_bonus

                # Verificar se completou (has_finished significa passou por TODOS os checkpoints)
                if kart.has_finished:
                    # Quando completa, next_checkpoint_id fica no último (3)
                    # mas na verdade passou por todos os 4 checkpoints (0,1,2,3)
                    evo_kart.checkpoints_reached = 4  # CORRIGIDO: 4 checkpoints total
                    evo_kart.completed = True  # Marca que completou!
                    evo_kart.is_alive = False

            # Desenhar
            if step % 2 == 0:  # A cada 2 frames para performance
                self._draw_generation(track, step, alive_count)
                self.clock.tick(self.fps)

            # Se todos morreram, terminar episódio
            if alive_count == 0:
                break

        # Calcular fitness de todos
        for evo_kart in self.population:
            evo_kart.calculate_fitness()

        return True

    def _draw_generation(self, track, step, alive_count):
        """Desenha o estado atual da geração."""
        self.screen.fill((0, 0, 0))

        # Desenhar pista
        for track_object in track.track_objects:
            track_object.draw(self.screen)

        # Desenhar karts vivos
        for evo_kart in self.population:
            if evo_kart.is_alive and not evo_kart.kart.has_finished:
                self._draw_colored_kart(evo_kart.kart, self.kart_colors[evo_kart.kart_id])

        # Informações
        y = 10
        info = [
            f"Geracao: {self.generation}",
            f"Passo: {step}/{self.max_steps_per_episode}",
            f"Karts vivos: {alive_count}/{self.population_size}",
            f"Melhor fitness anterior: {self.best_fitness_history[-1]:.1f}" if self.best_fitness_history else "Primeira geracao",
        ]

        for text_str in info:
            text = self.font.render(text_str, True, (255, 255, 255))
            self.screen.blit(text, (10, y))
            y += 22

        # Top 3 karts da geração anterior
        if self.generation > 0:
            y += 10
            text = self.font.render("Top 3 geracao anterior:", True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 22

            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for i in range(min(3, len(sorted_pop))):
                evo = sorted_pop[i]
                color = self.kart_colors[evo.kart_id]
                info_str = f"  #{evo.kart_id}: CPs={evo.checkpoints_reached}/4"
                text = self.font.render(info_str, True, color)
                self.screen.blit(text, (10, y))
                y += 20

        pygame.display.flip()

    def _draw_colored_kart(self, kart, color):
        """Desenha kart com cor específica."""
        import math

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

    def train(self, num_generations=50):
        """Treina por N gerações."""
        print("="*60)
        print("TREINAMENTO EVOLUTIVO INICIADO")
        print("="*60)
        print(f"População: {self.population_size} karts")
        print(f"Gerações: {num_generations}")
        print(f"Elite: {self.elite_size} melhores passam direto")
        print(f"Taxa de mutação: {self.mutation_rate}")
        print("="*60)

        for gen in range(num_generations):
            print(f"\nExecutando Geração {self.generation}...")

            if not self.run_generation():
                print("\nTreinamento interrompido pelo usuário")
                break

            self._create_next_generation()

        print("\n" + "="*60)
        print("TREINAMENTO EVOLUTIVO COMPLETO!")
        print("="*60)
        print(f"Gerações completadas: {self.generation}")
        print(f"Melhor fitness final: {self.best_fitness_history[-1]:.1f}")
        print(f"Melhor de todos: {max(self.best_fitness_history):.1f}")
        print("="*60)

        # Salvar melhor
        best = max(self.population, key=lambda x: x.fitness)
        best.controller.save("rl/models/evolution_best.pth")

        # Salvar métricas para visualização
        import json
        from datetime import datetime

        metrics = {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'generations': self.generation,
            'population_size': self.population_size,
            'timestamp': datetime.now().isoformat()
        }

        metrics_file = f"metrics/evolution_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Métricas salvas em: {metrics_file}")

        pygame.time.wait(3000)
        pygame.quit()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Treinamento evolutivo de karts')
    parser.add_argument('--track', type=str, default='pista1',
                        help=f'Pista para treinar (default: pista1, opções: {list(config.TRACKS.keys())})')
    parser.add_argument('--generations', type=int, default=100,
                        help='Número de gerações (default: 100)')
    parser.add_argument('--population', type=int, default=config.DEFAULT_POPULATION_SIZE,
                        help=f'Tamanho da população (default: {config.DEFAULT_POPULATION_SIZE})')
    parser.add_argument('--fps', type=int, default=config.DEFAULT_FPS,
                        help=f'FPS da visualização (default: {config.DEFAULT_FPS}, use {config.FAST_TRAINING_FPS}+ para rápido)')

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

    trainer = EvolutionaryTrainer(
        track_config['track_string'],
        track_config['initial_position'],
        track_config['initial_angle'],
        population_size=args.population
    )

    # Ajustar FPS
    trainer.clock = pygame.time.Clock()
    trainer.fps = args.fps

    print(f"\n{'='*60}")
    print(f"TREINAMENTO EVOLUTIVO - {track_config['name']}")
    print(f"{'='*60}")
    print(f"Pista: {args.track}")
    print(f"Gerações: {args.generations}")
    print(f"População: {args.population} karts")
    print(f"FPS: {args.fps}")
    print(f"{'='*60}\n")

    trainer.train(num_generations=args.generations)


if __name__ == '__main__':
    main()
