import math
import pygame
from game.elements.boost import Boost
from game.elements.grass import Grass
from game.elements.road import Road
from game.elements.checkpoint import Checkpoint
from heapq import heappush, heappop

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50
COST_MAPPING = {
    Road: 2,        # Reduzido: road é bom
    Checkpoint: 1,  # Checkpoints são ótimos
    Boost: 1,       # Boost continua barato
    Grass: 500      # Aumentado: evitar grama a todo custo
}

class AI():
    def __init__(self):
        self.kart = None
        self.graph = None
        self.checkpoints = None
        # Cache para otimização
        self.cached_path = None
        self.cached_checkpoint_id = None
        self.path_recalc_counter = 0

    def move(self, string):
        # Determinar qual checkpoint procurar
        if self.kart.next_checkpoint_id == 0:
            checkpoint_char = 'C'
        elif self.kart.next_checkpoint_id == 1:
            checkpoint_char = 'D'
        elif self.kart.next_checkpoint_id == 2:
            checkpoint_char = 'E'
        elif self.kart.next_checkpoint_id == 3:
            checkpoint_char = 'F'

        # Encontrar a posição do próximo checkpoint na grade
        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]
        for row_index in range(len(rows)):
            row = rows[row_index]
            for col_index in range(len(row)):
                track_char = row[col_index]
                if track_char == checkpoint_char:
                    break
            else:
                continue
            break

        # Calcular posição do checkpoint em pixels (centro da célula)
        next_checkpoint_position = [
            col_index * BLOCK_SIZE + BLOCK_SIZE // 2,
            row_index * BLOCK_SIZE + BLOCK_SIZE // 2
        ]

        # OTIMIZAÇÃO: Só recalcular path quando necessário
        need_recalc = False

        # Checkpoint mudou
        if self.cached_checkpoint_id != self.kart.next_checkpoint_id:
            need_recalc = True
            self.cached_checkpoint_id = self.kart.next_checkpoint_id

        # Path não existe ou está vazio
        elif self.cached_path is None or len(self.cached_path) <= 1:
            need_recalc = True

        # Recalcular periodicamente (a cada 10 frames) para correção de curso
        elif self.path_recalc_counter >= 10:
            need_recalc = True
            self.path_recalc_counter = 0

        self.path_recalc_counter += 1

        # Executar A* apenas quando necessário
        if need_recalc:
            path = self.a_star(string, self.kart.position, next_checkpoint_position, self.h, self.neighbors)
            self.cached_path = path
        else:
            path = self.cached_path

        if path and len(path) > 1:
            # OTIMIZAÇÃO: Lookahead adaptativo
            # Usar lookahead maior para pistas complexas com obstáculos
            lookahead_distance = min(8, len(path) - 1)  # Aumentado para 8 para melhor planejamento
            next_point = path[lookahead_distance]

            relative_x = next_point[0] - self.kart.position[0]
            relative_y = next_point[1] - self.kart.position[1]
            next_point_angle = math.atan2(relative_y, relative_x)
            relative_angle = (next_point_angle - self.kart.angle + math.pi) % (2 * math.pi) - math.pi

            right = [False, False, False, True]
            left = [False, False, True, False]
            forward = [True, False, False, False]

            if relative_angle > MAX_ANGLE_VELOCITY:
                command = right
            elif relative_angle < -MAX_ANGLE_VELOCITY:
                command = left
            else:
                command = forward

            key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
            keys = {key: command[i] for i, key in enumerate(key_list)}
            return keys
        else:
            no_movement = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}
            return no_movement


    def a_star(self, string, start, goal, h, neighbors):
        """
        A* Search Algorithm

        A* is an informed search algorithm that uses a combination of the cost to
        reach a node (goal), and an estimated cost (h) from the current node to
        the goal, to determine the priority of nodes in the search.

        Wikipedia page: https://en.wikipedia.org/wiki/A*_search_algorithm
        """
        # Converter start e goal de pixels para coordenadas de grade (row, col)
        start_row = int(start[1]) // BLOCK_SIZE
        start_col = int(start[0]) // BLOCK_SIZE
        goal_row = int(goal[1]) // BLOCK_SIZE
        goal_col = int(goal[0]) // BLOCK_SIZE

        start_grid = (start_row, start_col)
        goal_grid = (goal_row, goal_col)

        def reconstruct_path(cameFrom, current):
            total_path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                total_path.insert(0, current)
            # Converter caminho de coordenadas de grade para pixels (centro das células)
            pixel_path = []
            for row, col in total_path:
                pixel_x = col * BLOCK_SIZE + BLOCK_SIZE // 2
                pixel_y = row * BLOCK_SIZE + BLOCK_SIZE // 2
                pixel_path.append((pixel_x, pixel_y))
            return pixel_path

        openSet = [(0, start_grid)]
        cameFrom = {}
        gScore = {start_grid: 0}
        fScore = {start_grid: h(string, start_grid, goal_grid)}

        while openSet:
            _, current = heappop(openSet)
            if current == goal_grid:
                return reconstruct_path(cameFrom, current)

            for neighbor in neighbors(string, current):
                # Converter coordenadas de grade para pixels para obter o elemento da track
                neighbor_row, neighbor_col = neighbor
                pixel_x = neighbor_col * BLOCK_SIZE + BLOCK_SIZE // 2
                pixel_y = neighbor_row * BLOCK_SIZE + BLOCK_SIZE // 2
                neighbor_track_element = self.kart.get_track_element(string, pixel_x, pixel_y)[0]
                cost = COST_MAPPING.get(neighbor_track_element, 1000)  # default alto se não encontrar

                tentative_gScore = gScore[current] + cost
                if tentative_gScore < gScore.get(neighbor, float('inf')):
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = tentative_gScore + h(string, neighbor, goal_grid)
                    heappush(openSet, (fScore[neighbor], neighbor))

        return None

    def h(self, string, current, goal):
        # current e goal são coordenadas de grade (row, col)
        current_row, current_col = current
        goal_row, goal_col = goal

        # Distância Manhattan em coordenadas de grade
        manhattan_distance = abs(current_row - goal_row) + abs(current_col - goal_col)

        # Obter informações adicionais para a heurística
        speed_x, speed_y = self.kart.get_speed()
        kart_speed = math.sqrt(speed_x**2 + speed_y**2)

        # Converter coordenadas de grade para pixels para obter o elemento da track
        pixel_x = current_col * BLOCK_SIZE + BLOCK_SIZE // 2
        pixel_y = current_row * BLOCK_SIZE + BLOCK_SIZE // 2
        track_element, _ = self.kart.get_track_element(string, pixel_x, pixel_y)
        terrain_penalty = 1

        if track_element == Grass:
            terrain_penalty = 2

        combined_heuristic = manhattan_distance + terrain_penalty * kart_speed

        return combined_heuristic

    def neighbors(self, string, current):
        # current é (row, col) em coordenadas de grade
        row, col = current
        possible_neighbors = [
            (row, col + 1),      # direita
            (row, col - 1),      # esquerda
            (row + 1, col),      # baixo
            (row - 1, col),      # cima
            (row - 1, col - 1),  # diagonal cima-esquerda
            (row - 1, col + 1),  # diagonal cima-direita
            (row + 1, col - 1),  # diagonal baixo-esquerda
            (row + 1, col + 1)   # diagonal baixo-direita
        ]

        # Converter coordenadas de grade para pixels para validar com get_track_element
        valid_neighbors = []
        for n_row, n_col in possible_neighbors:
            # Converter para pixels (centro da célula)
            pixel_x = n_col * BLOCK_SIZE + BLOCK_SIZE // 2
            pixel_y = n_row * BLOCK_SIZE + BLOCK_SIZE // 2
            track_element = self.kart.get_track_element(string, pixel_x, pixel_y)[0]
            if track_element in (Road, Boost, Checkpoint, Grass):
                valid_neighbors.append((n_row, n_col))

        return valid_neighbors