import pygame
BLOCK_SIZE = 50

class Checkpoint():
    
    def __init__(self, x, y, checkpoint_id):
        self.rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
    
    def draw(self, screen):
        pygame.draw.rect(screen, (128, 128, 128), self.rect)