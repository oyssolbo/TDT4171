import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param


def backward(curr_idx: 'int', max_idx: 'int')->ndarray:
  if curr_idx > max_idx or curr_idx < 0:
    return np.ones_like(uw_param.x0)

  O = uw_param.H[uw_param.z[curr_idx]]
  FO = uw_param.F @ O

  if curr_idx > max_idx:
    return np.ones_like(uw_param.x0)
  if curr_idx == max_idx:
    return FO @ np.ones_like(uw_param.x0)
  return FO @ backward(curr_idx + 1, max_idx)

if __name__ == '__main__':
  xb = np.ones((uw_param.z.shape[0], 2))
  for idx, _ in enumerate(xb):
    xb[idx,:] = backward(idx, uw_param.z.shape[0] - 1).T

  plt.scatter(list(range(uw_param.z.shape[0])), xb[:,0])
  plt.ylim([0.0, 1.0])
  plt.title("Backward probability (not normalized)")
  plt.xlabel("Start-date of the backwards propagation")
  plt.ylabel("P(z_(k + 1:T) | x_k)")
  plt.show()

  # For just getting the raw data
  print(xb)