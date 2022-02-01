import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param


def backward(curr_idx: 'int', max_idx: 'int')->ndarray:
  # Unsure whether this should be normalized or not
  # Equation 14.13 in the book does not use normalization, however
  # it does not make sence to give probability that does not sum 
  # up to 1. I therefore normalize here 

  O = uw_param.H[uw_param.z[curr_idx]]
  FO = uw_param.F @ O

  if curr_idx == max_idx:
    prod = FO @ np.ones_like(uw_param.x0)
    return prod / prod.sum() 
  prod = FO @ backward(curr_idx + 1, max_idx)
  return prod / prod.sum()


if __name__ == '__main__':
  xb = np.ones((uw_param.z.shape[0], 2))
  for idx, _ in enumerate(xb):
    xb[idx,:] = backward(idx, uw_param.z.shape[0] - 1).T

  plt.scatter(list(range(uw_param.z.shape[0])), xb[:,0])
  plt.ylim([0.0, 1.0])
  plt.title("Backward probability (normalized)")
  plt.xlabel("Start-date of the backwards propagation")
  plt.ylabel("P(z_(t:T) | x_t)")
  plt.show()

  # For just getting the raw data
  print(xb)