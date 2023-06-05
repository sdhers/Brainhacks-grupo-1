import pandas as pd
import numpy as np

def smooth(vec, n):
    kernel = [1 / n] * n
    return np.convolve(vec, kernel, "same")

class Point:
    def __init__(self, df, table):
        data = df[table]
        
        x = smooth(data['x'], 5)
        y = smooth(data['y'], 5)
        
        self.positions = np.dstack((x, y))[0]
    
    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.positions - p2.positions, axis=1)
        
class Obj(Point):
    def __init__(self, df, table):
        super(Obj, self).__init__(df, table)
        
        self.positions = np.repeat(
            np.expand_dims(np.mean(self.positions, axis=0), axis=0),
            len(self.positions),
            axis=0
        )

class Vector():
    def __init__(self, p1, p2, normalize=True):
        self.positions = p2.positions - p1.positions
        
        self.norm = np.linalg.norm(self.positions, axis=1)
        
        if normalize:
            self.positions = self.positions / np.repeat(
                np.expand_dims(
                    self.norm,
                    axis=1
                ),
                2,
                axis=1
            )
            
    @staticmethod
    def cosine(v1, v2):
        length = len(v1.positions)
        cos = np.zeros(length)

        for i in range(length):
            cos[i] = np.dot(v1.positions[i], v2.positions[i])
            
        return cos
