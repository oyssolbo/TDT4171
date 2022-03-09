import numpy as np
import warnings

class DataNode:
  def __init__(
        self, 
        data      : np.ndarray,
        node_type : int,
        children  : list    = [],
        attribute : int     = None
      ) -> None:
    self.__data = data
    self.__node_type = node_type
    self.__children = children
    self.__attribute = attribute

    if node_type is not None and attribute is not None:
      warnings.warn("Incorrect combination of attribute and type set")

  def get_type(self) -> int:
    return self.__node_type

  def get_attribute(self) -> int: 
    return self.__attribute

  def get_data(self) -> np.ndarray:
    return self.__data

  def set_data(
        self, 
        data : np.ndarray
      ) -> None:
    self.__data = data

  def add_child(
        self, 
        child : 'DataNode',
        label : int
      ) -> None:
    self.__children.append((child, label))

  def get_children(self) -> list:
    return self.__children
    