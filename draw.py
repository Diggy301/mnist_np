import pygame
import math
import numpy as np


class drawingWindow():
    def __init__(self, size, block_size):
        self.size = size
        self.ncols = size
        self.nrows = size 
        self.block_size = block_size
        self.win = pygame.display.set_mode((size*block_size, size*block_size))
        self.grid = np.zeros((size, size))
        self.run = True


    def draw(self):
        self.win.fill((255,255,255)) # fill white

        for row in range(self.ncols):
            for col in range(self.nrows):
                if self.grid[row][col] == 1:
                    color = (0,0,0)
                else:
                    color = (255,255,255)
                
                pygame.draw.rect(self.win, color, pygame.Rect(row*self.block_size, col*self.block_size, self.block_size, self.block_size))
        #drawgrid
        for i in range(self.nrows):
            pygame.draw.line(self.win, (128,128,128), (0, i*20), (28*20, i*20))
            for j in range(self.ncols):
                pygame.draw.line(self.win, (128,128,128), (j*20, 0), (j*20, 28*20))


        pygame.display.update()


    def get_clicked_pos(self, pos):
        y, x = pos
        row = y // self.block_size
        col = x // self.block_size
        return row, col


    def mainloop(self):
        while self.run:
            self.draw()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
                if pygame.mouse.get_pressed()[0]: # LEFT
                    pos = pygame.mouse.get_pos()
                    row, col = self.get_clicked_pos(pos)
                    self.grid[row][col] = 1

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.run = False

        pygame.quit()
        return self.grid
