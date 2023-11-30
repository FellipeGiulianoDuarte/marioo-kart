import pygame

BLOCK_SIZE = 50


class Lava():

    def __init__(self, x, y):
        self.rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 0, 0), self.rect)
