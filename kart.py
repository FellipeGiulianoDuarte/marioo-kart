import math

import pygame

from kartPhysics import KartPhysics
from raceEventHandler import RaceEventHandler

from track import Track
from boost import Boost
from grass import Grass
from lava import Lava
from road import Road
from checkpoint import Checkpoint

MAX_ANGLE_VELOCITY = 0.05
MAX_ACCELERATION = 0.25
BLOCK_SIZE = 50
KART_RADIUS = 20
BOOST_VELOCITY = 25


class Kart(KartPhysics, RaceEventHandler):
    """
    Class implementing the display and physics of the kart in the game.

    Attributes:
        has_finished (bool): Flag indicating whether the kart has finished the race.
        controller: The controller object associated with the kart.
        position (tuple): Current position of the kart (x, y).
        last_position (tuple): Previous position of the kart (x, y).
        start_position (tuple): Initial position of the kart.
        start_orientation (float): Initial orientation of the kart.
        last_speed (tuple): Last recorded speed of the kart (delta_x, delta_y).
        angle (float): Current orientation angle of the kart.
        last_angle (float): Previous orientation angle of the kart.
        current_acceleration (float): Current acceleration of the kart.
        next_checkpoint_id (int): ID of the next checkpoint the kart should pass.
        orientation_from_checkpoint (float): Orientation angle recorded at the last checkpoint.
        position_from_checkpoint (tuple): Position recorded at the last checkpoint.
    """

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
        """
        Reset the kart to the specified initial position and orientation.

        Args:
            initial_position (tuple): The initial position of the kart (x, y).
            initial_orientation (float): The initial orientation angle of the kart.
        """
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
        # 0(t) = 0(t-1) - v0
        self.last_angle = self.angle
        self.angle -= MAX_ANGLE_VELOCITY

    def turn_right(self):
        # 0(t) = 0(t-1) + v0
        self.last_angle = self.angle
        self.angle += MAX_ANGLE_VELOCITY

    def get_track_element(self, string, x, y):
        rows = string.split('\n')
        rows = [row.strip() for row in rows if row.strip()]

        # Convert position indices to integers
        position_y = int(y)
        position_x = int(x)

        # Calculate row and column indices
        row_index = position_y // BLOCK_SIZE
        col_index = position_x // BLOCK_SIZE

        if 0 <= row_index < len(rows):
            row = rows[row_index]

            if 0 <= col_index < len(row):
                # Use the calculated indices to access the 2D array
                track_string = row[col_index]
                track_element = Track.char_to_track_element.get(track_string, None)
                if track_element is not None:
                    return track_element['class'], track_element.get('params', None)

        # Outside the track behaves like lava
        return Track.char_to_track_element['L']['class'], Track.char_to_track_element['L'].get('params', None)

    def update_position(self, string, screen):
        track_class, track_params = self.get_track_element(string, self.position[0], self.position[1])
        delta_x = self.position[0] - self.last_position[0]
        delta_y = self.position[1] - self.last_position[1]
        self.last_speed = (delta_x, delta_y)

        # Save the current position as the previous position
        self.last_position = self.position
        if track_class == Road:
            self.position = (self.calculate_x(0.02), self.calculate_y(0.02))
        elif track_class == Grass:
            self.position = (self.calculate_x(0.2), self.calculate_y(0.2))
        elif track_class == Boost:
            # Set velocity to 25 in formulas 3 and 4 for Boost
            self.position = (self.calculate_x(0.02, velocity=BOOST_VELOCITY), self.calculate_y(0.02, velocity=BOOST_VELOCITY))
        elif track_class == Checkpoint:
            self.position = (self.calculate_x(0.02), self.calculate_y(0.02))
            self.handle_checkpoint(track_params, string)
        elif track_class == Lava:
            self.handle_lava()

        # Set to zero in case was not used
        self.current_acceleration = 0

    def draw(self, screen):
        kart_position = [self.position[0], self.position[1]]

        # Draw the kart (circle)
        pygame.draw.circle(screen, (255, 255, 255), kart_position, KART_RADIUS)

        # Calculate vertices of the triangle
        triangle_size = 20  # Adjust the size of the triangle as needed
        angle_offset = math.pi / 2  # Offset to point the triangle in the right direction

        # Calculate the vertices of the triangle based on the angle
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
