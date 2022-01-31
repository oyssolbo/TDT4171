import numpy as np
from numpy import ndarray

# Using the models:
# x_{k+1} = Fx_{k}
# z_{k} = Hx_{k} 
F = np.array(
  [
    [0.7, 0.3],
    [0.3, 0.7]
  ]) 
H = np.array(
  [
    [0.9, 0],
    [0,   0.2]
  ])

# Initial value
x0 = np.array([[0.5], [0.5]])

# Observations
z = np.array([1, 1, 0, 1, 1])

# Just a cleaner version, while maintaining the comments/discussions I made
# above. SLightly modified to be more efficient and use DP
def forward(curr_idx: 'int', known_idx=0, known_value=x0)->float:
  if curr_idx == 0:
    return x0

  if known_idx != 0 and curr_idx == known_idx:
    sum = known_value
  else:
    sum = F @ forward(curr_idx - 1)
  
  prod = H @ sum
  return prod / prod.sum(axis=0)  

if __name__ == '__main__':
  # I have the problem that I don't use the values for z at all
  # This means that it will converge eventually, independent on the 
  # measurements for z

  # Looks like it just converges onto 0.9, since the values for z are not used
  xf = np.zeros((z.shape[0], 2))
  for idx, x in enumerate(xf):
    if idx == 0:
      known_idx = 0
    else:
      known_idx = idx - 1
    xf[idx,:] = forward(idx, known_idx, xf[known_idx]).T

  print(xf)





# def forward(curr_idx: 'int', known_idx=0, known_value=x0)->float:
#   if curr_idx == 0:
#     return x0
  
#   # But this is incorrect. I cannot understand how one should determine the
#   # value for H, since it will vary depending on the state of x. Since x is
#   # unknown, I cannot see how one should calulcate/determine this matrix. It 
#   # is wrong to just use the value for z, as it does not imply which  
#   if z[curr_idx] == 1:
#     O = H
#   else:
#     # And as expected, one gets a totally different result when using 1 - H
#     # as one should, as it is not correct
#     O = H #(1 - H)

#   # Initializing memory
#   sum = F @ forward(curr_idx - 1) #np.zeros((2, 1))

#   # Summing over array using recursion
#   # sum = F @ forward(idx - k - 1) #?
#   # for k in range(idx):
#   #   sum = sum + F @ forward(idx - k - 1)
#   prod = O @ sum
#   return prod / prod.sum(axis=0)  