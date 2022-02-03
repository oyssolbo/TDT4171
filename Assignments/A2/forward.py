import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray
from parameters import UmbrellaWorldParameters as uw_param

def forward(curr_idx: 'int', min_idx=0, known_idx=None, known_value=None)->ndarray:
  if curr_idx >= uw_param.z.shape[0] or curr_idx < min_idx:
    return np.zeros_like(uw_param.x0)
  
  O = uw_param.H[uw_param.z[curr_idx]]
  OFT = O @ uw_param.F.T

  if curr_idx == min_idx:
    prod = OFT @ uw_param.x0
    return prod / prod.sum(axis=0)

  # Start on DP
  if known_idx is not None and curr_idx is not None:
    val = known_value
  else:
    val = forward(curr_idx - 1)
  
  prod = OFT @ val
  return prod / prod.sum(axis=0)  

if __name__ == '__main__':
  xf = np.zeros((uw_param.z.shape[0], 2))
  for idx, x in enumerate(xf):
    if idx == 0:
      known_idx = 0
    else:
      known_idx = idx - 1
    xf[idx,:] = forward(idx, 0, known_idx, xf[known_idx]).T

  plt.scatter(list(range(uw_param.z.shape[0])), xf[:,0])
  plt.ylim([0.0, 1.0])
  plt.title("Forward probability for rain on a given day")
  plt.xlabel("Day")
  plt.ylabel("P(x | z)")
  plt.show()

  # For just getting the raw data
  print(xf)


