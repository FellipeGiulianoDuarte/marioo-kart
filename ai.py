import math
import pygame
from boost import Boost
from road import Road
from checkpoint import Checkpoint
from heapq import heappush, heappop

MAX_ANGLE_VELOCITY = 0.05
BLOCK_SIZE = 50

class AI():

    def __init__(self):
        self.kart = None
        self.graph = None
        self.checkpoints = None

    def move(self, string):
        """
        The AI uses A* pathfinding to navigate around obstacles (lava and grass) and reach checkpoints.

        :param string: The string describing the track
        :returns: A dictionary of keys (UP, DOWN, LEFT, RIGHT) and corresponding boolean values
        """

        # Find the position of the next checkpoint
        if self.kart.next_checkpoint_id == 0:
            char = 'C'
        elif self.kart.next_checkpoint_id == 1:
            char = 'D'
        elif self.kart.next_checkpoint_id == 2:
            char = 'E'
        elif self.kart.next_checkpoint_id == 3:
            char = 'F'

        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]

        for row_index in range(len(rows)):
            row = rows[row_index]

            for col_index in range(len(row)):
                track_char = row[col_index]
                
                if track_char == char:
                    break
            else:
                continue
            break

        next_checkpoint_position = [col_index * BLOCK_SIZE, row_index * BLOCK_SIZE + .5 * BLOCK_SIZE]
        print(next_checkpoint_position)

        # Use A* pathfinding to find the shortest path to the next checkpoint
        path = self.a_star(string, self.kart.position, next_checkpoint_position, self.h, self.neighbors)
        # Determine the angle and movement based on the path
        if path:
            next_point = path[1]
            print(path[1])
            relative_x = next_point[0] - self.kart.position[0]
            relative_y = next_point[1] - self.kart.position[1]
            next_point_angle = math.atan2(relative_y, relative_x)
            relative_angle = (next_point_angle - self.kart.angle + math.pi) % (2 * math.pi) - math.pi

            # Adjust movement based on the relative angle
            if relative_angle > MAX_ANGLE_VELOCITY:
                command = [False, False, False, True]  # Turn right
            elif relative_angle < -MAX_ANGLE_VELOCITY:
                command = [False, False, True, False]  # Turn left
            else:
                command = [True, False, False, False]  # Move forward

            key_list = [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]
            keys = {key: command[i] for i, key in enumerate(key_list)}
            return keys
        else:
            # If no path is found, stop moving
            return {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_LEFT: False, pygame.K_RIGHT: False}


    def a_star(self, string, start, goal, h, neighbors):
        def reconstruct_path(cameFrom, current):
            total_path = [current]
            while current in cameFrom:
                current = cameFrom[current]
                total_path.insert(0, current)
            return total_path
        
        # The set of discovered nodes that may need to be (re-)expanded.
        # Initially, only the start node is known.
        # This is usually implemented as a min-heap or priority queue rather than a hash-set.
        openSet = [(0, start)]

        # For node n, cameFrom[n] is the node immediately preceding it on the cheapest path from the start
        # to n currently known.
        cameFrom = {}

        # For node n, gScore[n] is the cost of the cheapest path from start to n currently known.
        gScore = {tuple(start): 0}

        # For node n, fScore[n] := gScore[n] + h(n). fScore[n] represents our current best guess as to
        # how cheap a path could be from start to finish if it goes through n.
        fScore = {tuple(start): h(start, goal)}

        while openSet:
            _, current = heappop(openSet)
            if tuple(current) == tuple(goal):
                return reconstruct_path(cameFrom, tuple(current))

            for neighbor in neighbors(string, current):
                tentative_gScore = gScore[tuple(current)] + 1
                if tentative_gScore < gScore.get(tuple(neighbor), float('inf')):
                    # This path to neighbor is better than any previous one. Record it!
                    cameFrom[tuple(neighbor)] = tuple(current)
                    gScore[tuple(neighbor)] = tentative_gScore
                    fScore[tuple(neighbor)] = tentative_gScore + h(neighbor, goal)
                    heappush(openSet, (fScore[tuple(neighbor)], tuple(neighbor)))

        # Open set is empty but goal was never reached
        return None  # failure

    def h(self, current, goal):
        return abs(int(current[0]) - int(goal[0])) + abs(int(current[1]) - int(goal[1]))

    def neighbors(self, string, current):
        x, y = current
        possible_neighbors = [(int(x + 1), int(y)), (int(x - 1), int(y)), (int(x), int(y + 1)), (int(x), int(y - 1))]
        
        # Filter valid neighbors based on track elements
        valid_neighbors = [
            (nx, ny) for nx, ny in possible_neighbors
            if self.kart.get_track_element(string, nx, ny)[0] in (Road, Boost, Checkpoint)
        ]
        return valid_neighbors
