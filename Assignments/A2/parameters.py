from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from numpy import ndarray

@dataclass
class UmbrellaWorldParameters:
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
  H_given_no_rain = np.eye(2) - H_given_rain
  H = np.array([H_given_no_rain, H_given_rain])

  # Initial value
  x0 = np.array([[0.5], [0.5]])

  # Observations
  z = np.array([1, 1, 0, 1, 1])