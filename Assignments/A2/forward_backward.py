import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param
from forward import forward
from backward import backward

def forward_backward():
  # Initiating memory
  xf = np.zeros((uw_param.z.shape[0], 2))
  for i in range(uw_param.x0.shape[1]):
    xf[0,i] = uw_param.x0[i]

  xs = np.zeros_like(xf) 
  xb = np.ones_like(xf)

  for idx, _ in enumerate(xf):
    # Calculating forward and backward pass
    xf[idx,:] = forward(idx).T
    xb[idx,:] = backward(idx + 1, uw_param.z.shape[0] - 1).T

    xf_xb = np.array([xf[idx, 0] * xb[idx, 0], xf[idx, 1] * xb[idx, 1]])
    xs[idx] = xf_xb / xf_xb.sum(axis=0)

  return xs

if __name__ == '__main__':
  xs = forward_backward()
  print(xs)

"""
Something is a bit wrong with my method, for the first couple of iterations.
By comparing the values to the expected results at 

https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

one can see that there is a disagreement for the first value. I cannot 
understand why this is only for the first value, and not the other values. 
"""

