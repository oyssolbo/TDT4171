import os
import sys
from attr import attrib

import numpy as np
import graphviz
import csv

import importance
import nodes

from typing import Callable, TextIO
from numpy import ndarray

class DecisionTree:
  def __init__(
        self,
        train_data_csv          : TextIO,
        test_data_csv           : TextIO,
      ) -> None:

    self.training_nodes = self.extract_data(train_data_csv)
    self.testing_nodes = self.extract_data(test_data_csv)

  def train_decision_tree(
        self,
        importance_func : Callable
      ) -> None:
    """
    Interface for training the decision tree 
    """

    def plurarity_node(
          data_node_list : list
        ) -> nodes.DataNode:
      """
      Extracts the plurarity-value of a set of examples, and determines
      the most common class/node based on said examples
      """
      if not data_node_list:
        data_node = nodes.DataNode(
          data=np.empty_like(data_node_list[0].get_data()),
          node_type=None,
          children=[]
        )
      else:
        # Create an array of the different classes
        class_arr = np.zeros((1, (len(data_node_list)))).reshape((1, -1))
        for (idx, node) in enumerate(data_node_list):
          class_arr[0, idx] = node.get_type()

        # Find the type that occurs most often
        counts = np.bincount(class_arr)
        most_common_type = np.argmax(counts)

        # Create a node with correct type
        data_node = nodes.DataNode(
          data=np.empty_like(data_node_list[0].get_data()),
          node_type=most_common_type,
          children=[]
        )
      print(data_node)
      return data_node

    def check_classification(
          data_node_list : list = []
        ) -> tuple:
      """
      Checks the classification of the data-nodes, and determines if they 
      all have the same type
      """
      if not data_node_list:
        return (False, None)

      # Create array
      class_arr = np.zeros((1, (len(data_node_list)))).reshape((1, -1))
      for (idx, node) in enumerate(data_node_list):
        class_arr[0, idx] = node.get_type()
      print(class_arr)

      if not np.all(class_arr == class_arr[0,0]):
        return (False, None) # Last argument contains no information in this case

      # Create a node with correct type
      data_node = nodes.DataNode(
        data=np.empty_like(data_node_list[0].get_data()),
        node_type=class_arr[0],
        children=[]
      )
      return (True, data_node)  
      
    def learn_decision_tree(
          importance_func         : Callable,
          current_data_node_list  : list = [],
          prev_data_node_list     : list = []
        ) -> nodes.DataNode:
      """
      Implements the decision-tree algorithm with pseudocode shown in 
      figure 19.5 in Russel and Norvig
      """
      has_common_class, common_node = check_classification(current_data_node_list)

      if not current_data_node_list:
        # examples-list is empty
        print("not current_data_node_list")
        return plurarity_node(prev_data_node_list)
      elif has_common_class:
        # SO occurs here!
        print("has_common_class")
        return common_node
      elif current_data_node_list[0].get_data().reshape((1, -1)).shape[1] == 0:
        # attributes-list is empty
        print("not current_data_node_list[0].get_data()")
        return plurarity_node(current_data_node_list)
      else:
        # Calculate the node with the best attribute
        # attribute = importance_func(current_data_node_list)
        attribute = importance_func(current_data_node_list)

        root_node = nodes.DataNode(
          data=np.empty_like(current_data_node_list[0].get_data()),
          node_type=None,
          children=[]
        )

        # Using prior knowledge that the minimum data is 1 and the maximum
        # data is 2
        min_val = 1
        max_val = 2
        vals = range(min_val, max_val + 1)

        next_node_list = []

        for val in vals: 
          for node in current_data_node_list:
            # Extract the nodes that match the 
            node_data = node.get_data()
            if node_data[attribute] == val:
              # Add to array for further invokations
              next_node_list.append(node)
          
          # Next set of recursion
          root_node.add_child(
            child=learn_decision_tree(
              importance_func=importance_func,
              current_data_node_list=next_node_list,
              prev_data_node_list=current_data_node_list
            ),
            label=val
          )

        # Return the root of the subtree
        # print(root_node)
        return root_node
    
    learn_decision_tree(
      importance_func=importance_func,
      current_data_node_list=self.testing_nodes, 
      prev_data_node_list=[]
    )


  def test_decision_tree(self) -> None:
    pass

  def extract_data(
        self,
        data_csv : TextIO,
      ) -> list:
    """
    Extracts the data from a csv_file and creates a set of attributes
    that are returned
    """
    data_node_list = []
    with open(data_csv) as csvfile:
      read_line = csv.reader(csvfile)
      for row in read_line:
        data = np.array(list(map(int, row)))
        node_type = data[-1]
        node = nodes.DataNode(data[:-1], node_type)
        data_node_list.append(node)
    return data_node_list

if __name__ == '__main__':
  train_data = os.path.join(sys.path[0], 'data/train.csv')
  test_data = os.path.join(sys.path[0], 'data/test.csv')

  importance_class = importance.Importance()
  random_importance_func = lambda x : importance_class.random(x)

  random_tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )
  random_tree.train_decision_tree(importance_func=random_importance_func)



