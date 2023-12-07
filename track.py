from grass import Grass
from checkpoint import Checkpoint
from boost import Boost
from lava import Lava
from road import Road

import pygame

BLOCK_SIZE = 50
BACKGROUND_COLOR = (0, 0, 0)

class Track(object):

    char_to_track_element = {
        'G': {
            'class': Grass,
            'params': []
        },
        'B': {
            'class': Boost,
            'params': []
        },
        'C': {
            'class': Checkpoint,
            'params': [0]
        },
        'D': {
            'class': Checkpoint,
            'params': [1]
        },
        'E': {
            'class': Checkpoint,
            'params': [2]
        },
        'F': {
            'class': Checkpoint,
            'params': [3]
        },
        'L': {
            'class': Lava,
            'params': []
        },
        'R': {
            'class': Road,
            'params': []
        }
    }

    def __init__(self, string, initial_position, initial_angle):
        self.string = string

        self.__initial_position = initial_position
        self.__initial_angle = initial_angle

        self.track_objects, self.width, self.height = self.parse_string(string)

        self.__karts = []

    @property
    def initial_position(self):
        return self.__initial_position

    @property
    def initial_angle(self):
        return self.__initial_angle

    @property
    def karts(self):
        return self.__karts

    def add_kart(self, kart):
        self.__karts.append(kart)

    def parse_string(self, string):

        track_objects = []
        width = 0
        height = 0

        x = 0
        y = 0
        for c in string:
            if c in Track.char_to_track_element.keys():
                track_element = Track.char_to_track_element[c]
                track_class = track_element['class']
                track_params = [x, y] + track_element['params']
                track_objects.append(track_class(*track_params))
                x += BLOCK_SIZE
                width += BLOCK_SIZE
            elif c == '\n':
                x = 0
                y += BLOCK_SIZE
                width = 0
                height += BLOCK_SIZE
        height += BLOCK_SIZE
        return track_objects, width, height

    def play(self):

        pygame.init()

        screen = pygame.display.set_mode((self.width, self.height))

        for kart in self.karts:
            kart.reset(self.initial_position, self.initial_angle)

        running = True
        cycles = 0
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill(BACKGROUND_COLOR)

            for track_object in self.track_objects:
                track_object.draw(screen)

            for kart in self.karts:

                keys = kart.controller.move(self.string)

                if keys[pygame.K_UP]:
                    kart.forward()
                if keys[pygame.K_DOWN]:
                    kart.backward()
                if keys[pygame.K_LEFT]:
                    kart.turn_left()
                if keys[pygame.K_RIGHT]:
                    kart.turn_right()

                kart.update_position(self.string, screen)

                if not kart.has_finished:
                    kart.draw(screen)

            if all([k.has_finished for k in self.karts]):
                running = False

            pygame.display.flip()

            cycles += 1

        print("Finished in", cycles, "cycles!")

        pygame.quit()

        return cycles