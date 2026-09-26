import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        z = np.subtract(z, np.max(z))
        denom = np.sum(np.exp(z))
        sm = np.array([np.exp(i) / denom for i in z])

        return np.round(sm, 4)
