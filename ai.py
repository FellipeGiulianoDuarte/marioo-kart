import math
import pygame
from boost import Boost
from grass import Grass
from road import Road
from checkpoint import Checkpoint
from heapq import heappush, heappop

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50
COST_MAPPING = {
    Road: 3,
    Checkpoint: 3,
    Boost: 1,
    Grass: 300
}

class AI():
    def __init__(self):
        self.kart = None
        self.graph = None
        self.checkpoints = None

    def move(self, string):

        if self.kart.next_checkpoint_id == 0:
            checkpoint_char = 'C'
        elif self.kart.next_checkpoint_id == 1:
            checkpoint_char = 'D'
        elif self.kart.next_checkpoint_id == 2:
            checkpoint_char = 'E'
        elif self.kart.next_checkpoint_id == 3:
            checkpoint_char = 'F'
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

        next_checkpoint_position = [col_index * BLOCK_SIZE, row_index * BLOCK_SIZE + .5 * BLOCK_SIZE]

        path = self.a_star(string, self.kart.position, next_checkpoint_position, self.h, self.neighbors)

        if path:
            next_point = path[1]
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
        def reconstruct_path(cameFrom, current):
            total_path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                total_path.insert(0, current)
            return total_path

        openSet = [(0, start)]
        cameFrom = {}
        gScore = {tuple(start): 0}
        fScore = {tuple(start): h(string, start, goal)}

        while openSet:
            _, current = heappop(openSet)
            if tuple(current) == tuple(goal):
                return reconstruct_path(cameFrom, tuple(current))

            for neighbor in neighbors(string, current):
                # linhas com problemas
                neighbor_track_element = self.kart.get_track_element(string, neighbor[0], neighbor[1])[0]
                cost = COST_MAPPING.get(neighbor_track_element)
                tentative_gScore = gScore[tuple(current)] + cost
                # acabou aqui
                if tentative_gScore < gScore.get(tuple(neighbor), float('inf')):
                    cameFrom[tuple(neighbor)] = tuple(current)
                    gScore[tuple(neighbor)] = tentative_gScore
                    fScore[tuple(neighbor)] = tentative_gScore + h(string, neighbor, goal)
                    heappush(openSet, (fScore[tuple(neighbor)], tuple(neighbor)))

        return None

    def h(self, string, current, goal):

        manhattan_distance = abs(int(current[0]) - int(goal[0])) + abs(int(current[1]) - int(goal[1]))
        kart_speed = self.kart.get_speed()

        speed_x, speed_y = self.kart.get_speed()
        kart_speed = math.sqrt(speed_x**2 + speed_y**2)
        track_element, _ = self.kart.get_track_element(string, current[0], current[1])
        terrain_penalty = 1

        if track_element == Grass:
            terrain_penalty = 2

        combined_heuristic = manhattan_distance + terrain_penalty * kart_speed

        return combined_heuristic

    def neighbors(self, string, current):
        x, y = current
        possible_neighbors = [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x - 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
            (x + 1, y + 1)
        ]

        valid_neighbors = [
            (nx, ny) for nx, ny in possible_neighbors
            if self.kart.get_track_element(string, nx, ny)[0] in (Road, Boost, Checkpoint, Grass)
        ]
        return valid_neighbors