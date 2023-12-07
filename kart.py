import math
import pygame

from boost import Boost
from checkpoint import Checkpoint
from grass import Grass
from lava import Lava
from road import Road
from track import Track

from kartPhysics import KartPhysics
from raceEventHandler import RaceEventHandler


MAX_ANGLE_VELOCITY = 0.05
MAX_ACCELERATION = 0.25
BLOCK_SIZE = 50
KART_RADIUS = 20
BOOST_VELOCITY = 25


class Kart(KartPhysics, RaceEventHandler):

    def __init__(self, controller):
        super().__init__()
        self.has_finished = False
        self.controller = controller
        self.controller.kart = self

        self.next_checkpoint_id = 0

        self.position = (0, 0)
        self.last_position = (0, 0)

        self.start_position = (0, 0)
        self.start_orientation = 0

        self.angle = 0
        self.last_angle = 0

    def reset(self, initial_position, initial_orientation):

        self.last_position = initial_position
        self.position = initial_position

        self.last_angle = initial_orientation
        self.angle = initial_orientation

        self.position_from_checkpoint = initial_position
        self.orientation_from_checkpoint = initial_orientation

    def forward(self):
        if self.current_acceleration <= 0:
            self.current_acceleration += MAX_ACCELERATION

    def backward(self):
        if self.current_acceleration >= 0:
            self.current_acceleration -= MAX_ACCELERATION

    def turn_left(self):
        self.last_angle = self.angle
        self.angle -= MAX_ANGLE_VELOCITY

    def turn_right(self):
        self.last_angle = self.angle
        self.angle += MAX_ANGLE_VELOCITY

    def get_speed(self):
        return self.last_speed

    def get_track_element(self, string, x, y):
        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]

        position_y = int(y)
        position_x = int(x)

        row_index = position_y // BLOCK_SIZE
        col_index = position_x // BLOCK_SIZE

        if 0 <= row_index < len(rows):
            row = rows[row_index]

            if 0 <= col_index < len(row):
                track_string = row[col_index]
                track_element = Track.char_to_track_element.get(track_string, None)
                if track_element is not None:
                    return track_element['class'], track_element.get('params', None)

        return Track.char_to_track_element['L']['class'], Track.char_to_track_element['L'].get('params', None)

    def update_position(self, string, screen):
        track_class, track_params = self.get_track_element(string, self.position[0], self.position[1])
        delta_x = self.position[0] - self.last_position[0]
        delta_y = self.position[1] - self.last_position[1]
        self.last_speed = (delta_x, delta_y)

        self.last_position = self.position
        if track_class == Road:
            self.position = (self.calculate_x(0.02), self.calculate_y(0.02))
        elif track_class == Grass:
            self.position = (self.calculate_x(0.2), self.calculate_y(0.2))
        elif track_class == Boost:
            self.position = (self.calculate_x(0.02, velocity=BOOST_VELOCITY), self.calculate_y(0.02, velocity=BOOST_VELOCITY))
        elif track_class == Checkpoint:
            self.position = (self.calculate_x(0.02), self.calculate_y(0.02))
            self.handle_checkpoint(track_params, string)
        elif track_class == Lava:
            self.handle_lava()

        self.current_acceleration = 0

    def draw(self, screen):
        kart_position = [self.position[0], self.position[1]]

        pygame.draw.circle(screen, (255, 255, 255), kart_position, KART_RADIUS)

        triangle_size = 20
        angle_offset = math.pi / 2

        vertices = [
            (
                kart_position[0] + int(triangle_size *
                                       math.cos(self.angle + angle_offset)),
                kart_position[1] + int(triangle_size *
                                       math.sin(self.angle + angle_offset))
            ),
            (
                kart_position[0] + int(triangle_size *
                                       math.cos(self.angle - angle_offset)),
                kart_position[1] + int(triangle_size *
                                       math.sin(self.angle - angle_offset))
            ),
            (
                kart_position[0] + int(triangle_size *
                                       1.5 * math.cos(self.angle)),
                kart_position[1] + int(triangle_size *
                                       1.5 * math.sin(self.angle))
            )
        ]

        pygame.draw.polygon(screen, (255, 255, 255), vertices)
