import numpy as np
from numpy.typing import NDArray
import math

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        
        return [round(float(1/(1 + math.e ** -i)), 5) for i in z]

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        
        return [round(float(max(0, i)), 5) for i in z]
