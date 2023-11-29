import math
import pygame
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


class Kart():  # Vous pouvez ajouter des classes parentes
    """
    Classe implementant l'affichage et la physique du kart dans le jeu
    """

    def __init__(self, controller):
        self.has_finished = False
        self.controller = controller
        self.controller.kart = self

        self.position = (0, 0)
        self.last_position = (0, 0)

        self.start_position = (0, 0)
        self.start_orientation = 0

        self.last_speed = (0, 0)

        self.angle = 0
        self.last_angle = 0

        self.current_acceleration = 0

        self.next_checkpoint_id = 0

        self.orientation_from_checkpoint = 0
        self.position_from_checkpoint = (0, 0)

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

        # Outside of the track behaves like lava
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

    def handle_checkpoint(self, checkpoint_params, string):
        checkpoint_id = checkpoint_params[0]
        last_checkpoint_id = self.get_last_checkpoint_id(string)
        
        if checkpoint_id == self.next_checkpoint_id:
            # Last checkpoint
            if checkpoint_id == last_checkpoint_id:
                self.has_finished = True
                return

            # Save the checkpoint, but this is not the last one
            self.position_from_checkpoint = self.position
            self.orientation_from_checkpoint = self.angle
            self.next_checkpoint_id += 1

    def handle_lava(self):
        # Reset from the last checkpoint
        self.last_position = self.position_from_checkpoint
        self.position = self.position_from_checkpoint
        self.last_angle = self.orientation_from_checkpoint
        self.angle = self.orientation_from_checkpoint

    # Returns last checkpoint ID on the track
    def get_last_checkpoint_id(self, string):
        letters = {'C', 'D', 'E', 'F'}
        count = 0
        seen_letters = set()

        for char in string:
            if char in letters and char not in seen_letters:
                count += 1
                seen_letters.add(char)

        return count - 1

    # formula 1 0v(t − 1) = arctan (Vy(t-1)/Vx(t-1))
    def update_angle(self):
        return math.atan2(self.last_speed[1], self.last_speed[0])

    # formula 2 |v(t − 1)| = q Vx(t − 1)2 + vy(t − 1)2
    def update_speed(self):
        return math.sqrt(self.last_speed[0] ** 2 + self.last_speed[1] ** 2)

    # formula 3 a(t) = ac(t) − f ∗ |v(t − 1)| ∗ cos(0(t) - 0v(t-1))
    def calculate_current_acceleration(self, friction):
        last_speed = self.update_speed()
        prev_angle = self.update_angle()
        current_accel = self.current_acceleration - friction * last_speed * math.cos(self.angle - prev_angle)
        return current_accel

    # formula 4 v(t) = a(t) + v(t − 1)
    def calculate_current_speed(self, friction):
        last_speed = self.update_speed()
        current_acceleration = self.calculate_current_acceleration(friction)
        return current_acceleration + last_speed

    # formula 7 x(t) = x(t − 1) + vx(t)
    def calculate_x(self, friction, velocity=None):
        if velocity is None:
            current_speed = self.calculate_current_speed(friction)
        else:
            current_speed = velocity

        speed_x = current_speed * math.cos(self.angle)

        self.last_speed = (speed_x, self.last_speed[1])
        return self.last_position[0] + speed_x

    # formula 8 y(t) = y(t − 1) + vy(t)
    def calculate_y(self, friction, velocity=None):
        if velocity is None:
            current_speed = self.calculate_current_speed(friction)
        else:
            current_speed = velocity

        speed_y = current_speed * math.sin(self.angle)

        self.last_speed = (self.last_speed[0], speed_y)
        return self.last_position[1] + speed_y

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
