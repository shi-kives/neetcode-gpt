import numpy as np
from numpy.typing import NDArray

class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        
        output = [np.round(float(1/(1 + np.e ** -i)), 5) for i in z]
        return output

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        
        output = [np.round(float(max(0, i)), 5) for i in z]
        return output
