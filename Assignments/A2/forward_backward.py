import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param
from forward import forward
from backward import backward

def forward_backward(min_idx: 'int', max_idx: 'int')->'ndarray':
  if min_idx > max_idx or min_idx < 0 or max_idx >= uw_param.z.shape[0]:
    return np.empty((uw_param.z.shape[0], 2))

  # Initiating memory
  xf = np.zeros((max_idx+1, 2))
  for i in range(uw_param.x0.shape[1]):
    xf[0,i] = uw_param.x0[i]

  xs = np.zeros_like(xf) 
  xb = np.ones_like(xf)

  for idx, _ in enumerate(xf):
    # Calculating forward and backward pass
    xf[idx,:] = forward(curr_idx=idx, min_idx=min_idx).T
    xb[idx,:] = backward(curr_idx=idx+1, max_idx=max_idx).T
    print((idx + 2, ":", max_idx + 1, xb[idx,:]))

    xf_xb = np.array([xf[idx, i] * xb[idx, i] for i in range(xf.shape[1])]).T 
    xs[idx] = xf_xb / xf_xb.sum(axis=0)

  return xs

if __name__ == '__main__':
  xs = forward_backward(min_idx=0, max_idx=1)
  print(xs)

  xs = forward_backward(min_idx=0, max_idx=uw_param.z.shape[0]-1)
  print(xs)

"""
Something is a bit wrong with my method, for the first couple of iterations.
By comparing the values to the expected results at 

https://en.wikipedia.org/wiki/Forward%E2%80%93backward_algorithm

one can see that there is a disagreement for the first value. I cannot 
understand why this is only for the first value, and not the other values. 
"""

