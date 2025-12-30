import pygame

BLOCK_SIZE = 50


class TrackElement:

    def __init__(self, x, y, color):
        self.rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
        self.color = color

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)