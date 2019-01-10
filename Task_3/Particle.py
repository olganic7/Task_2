#!/usr/bin/env python
from random import uniform
import math
class Coord:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
class Speed:
    def __init__(self, u, v, w):
        self.u = u
        self.v = v
        self.w = w

class Particle:
 
    # initialise
    def __init__(self, coord, speed, weight, col):
 
        # active settings
        self.is_active = True
        self.life = 0.0
        self.ageing = 0.0
 
        # color
        self.color = col
 
        # coordinates
        self.x = coord.x
        self.y = coord.y
        self.z = coord.z
 
        # velocity
        self.xv = speed.u
        self.yv = speed.v
        self.zv = speed.w
        
        # weight
        self.m = weight