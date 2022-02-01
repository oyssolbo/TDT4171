import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param
from forward import forward
from backward import backward

def forward_backward():
  # Initiating memory
  xf = np.zeros((uw_param.z.shape[0], 2))
  for i in range(2):
    xf[0,i] = uw_param.x0[i]

  xs = np.zeros_like(xf) 
  xb = np.ones_like(xf)

  # Calculating the forward pass
  for idx, _ in enumerate(xf):
    if idx == 0:
      known_idx = 0
    else:
      known_idx = idx - 1
    xf[idx,:] = forward(idx, known_idx, xf[known_idx]).T
  
  # Calculating the backward pass
  for idx, _ in enumerate(xb):
    xb[idx,:] = backward(idx, uw_param.z.shape[0] - 1).T

  # Calculating the backwards pass
  for i in range(xs.shape[0] - 1, -1, -1):
    xf_xb = np.array([xf[i, 0] * xb[i, 0], xf[i, 1] * xb[i, 1]])
    xs[i] = xf_xb / xf_xb.sum(axis=0)

  return xs

if __name__ == '__main__':
  xs = forward_backward()
  print(xs)

