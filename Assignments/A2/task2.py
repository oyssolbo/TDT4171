import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

# Using the models:
# x_{k+1} = Fx_{k}
# z_{k} = Hx_{k} 
F = np.array(
  [
    [0.7, 0.3],
    [0.3, 0.7]
  ]) 
H_given_rain = np.array(
  [
    [0.9, 0],
    [0,   0.2]
  ])
H_given_no_rain = 1 - H_given_rain
H = np.array([H_given_no_rain, H_given_rain])

# Initial value
x0 = np.array([[0.5], [0.5]])

# Observations
z = np.array([1, 1, 0, 1, 1])

def forward(curr_idx: 'int', known_idx=0, known_value=x0)->ndarray:
  if curr_idx == 0:
    return x0

  if known_idx != 0 and curr_idx == known_idx:
    sum = known_value
  else:
    sum = F @ forward(curr_idx - 1)
  
  O = H[z[idx]]
  prod = O @ sum
  return prod / prod.sum(axis=0)  

if __name__ == '__main__':
  xf = np.zeros((z.shape[0], 2))
  for idx, x in enumerate(xf):
    if idx == 0:
      known_idx = 0
    else:
      known_idx = idx - 1
    xf[idx,:] = forward(idx, known_idx, xf[known_idx]).T

  plt.scatter(list(range(z.shape[0])), xf[:,0])
  plt.ylim([0.0, 1.0])
  plt.title("Probability for rain on a given day")
  plt.xlabel("Day")
  plt.ylabel("P(x | z)")
  plt.show()

  # For just getting the raw data
  print(xf)


