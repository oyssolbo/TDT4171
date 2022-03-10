import os
import sys

import numpy as np
import graphviz
import csv
import warnings
import copy

import importance
import nodes

import matplotlib.pyplot as plt
import math

from typing import Callable, TextIO

class DecisionTree:
  def __init__(
        self,
        train_data_csv  : TextIO,
        test_data_csv   : TextIO,
      ) -> None:

    self.__training_nodes = self.extract_data(train_data_csv)
    self.__testing_nodes = self.extract_data(test_data_csv)

    self.__root_node = None

    # Using prior knowledge that the minimum data is 1 and the maximum
    # data is 2
    self.__min_val = 1
    self.__max_val = 2

    # Showing the depth of the tree
    self.__tree_depth = 0

    # Keeping track of which attribute has been removed. Otherwise, the 
    self.__num_training_attributes = self.__training_nodes[0].get_data().shape[0]
    self.__training_attributes_list = [*range(0, self.__num_training_attributes, 1)]


  def train_decision_tree(
        self,
        importance_func : Callable
      ) -> None:
    """
    Interface for training the decision tree 
    """

    def plurarity_node(
          data_node_list : list#[nodes.DataNode]
        ) -> nodes.DataNode:
      """
      Extracts the plurarity-value of a set of examples, and determines
      the most common class/node based on said examples
      """
      if not data_node_list:
        data_node = nodes.DataNode(
          data=np.array([]), 
          node_type=None,
          children=[]
        )
      else:
        # Create an array of the different classes
        class_arr = np.zeros(len(data_node_list), dtype=int).reshape((1, -1))
        for (idx, node) in enumerate(data_node_list):
          class_arr[0, idx] = node.get_type()

        # Find the type that occurs most often
        counts = np.bincount(class_arr[0])
        most_common_type = np.argmax(counts)

        # Create a node with correct type
        data_node = nodes.DataNode(
          data=np.empty_like(data_node_list[0].get_data()),
          node_type=most_common_type,
          children=[]
        )
      return data_node

    def check_classification(
          data_node_list : list = [] #[nodes.DataNode] = []
        ) -> tuple: #[bool, nodes.DataNode]:
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

      # Checking if all elements are equal
      if not np.all(class_arr == class_arr[0,0]):
        return (False, None) # Last argument contains no information in this case

      # Create a node with correct type, since all have the same type
      data_node = nodes.DataNode(
        data=np.empty_like(data_node_list[0].get_data()),
        node_type=class_arr[0,0],
        children=[]
      )
      return (True, data_node)  
      
    def learn_decision_tree(
          importance_func         : Callable,
          current_data_node_list  : list      = [],
          prev_data_node_list     : list      = [],
          available_attributes    : list      = [],
          tree_depth              : int       = 0
        ) -> nodes.DataNode:
      """
      Implements the decision-tree algorithm with pseudocode shown in 
      figure 19.5 in Russel and Norvig
      """
      tree_depth += 1
      if tree_depth >= self.__tree_depth:
        self.__tree_depth = tree_depth 

      has_common_class, common_node = check_classification(current_data_node_list)

      if not current_data_node_list:
        # examples-list is empty
        # print("not current_data_node_list")
        return plurarity_node(prev_data_node_list)
      elif has_common_class:
        # print("has_common_class")
        return common_node
      elif current_data_node_list[0].get_data().reshape((1, -1)).shape[1] == 0:
        # attributes-list is empty
        # print("not current_data_node_list[0].get_data()")
        return plurarity_node(current_data_node_list)
      else:
        # Check that all nodes have a similar size of the data-array
        # This is no longer necessary, as the problem was solved
        current_data_shape = (-1,-1)
        for node in current_data_node_list:
          if current_data_shape == (-1,-1):
            current_data_shape = node.get_data().shape
            continue
          if current_data_shape != node.get_data().shape:
            warnings.warn("Incorrect shape between nodes!")
            quit()

        # Calculate the node with the best attribute
        # The local attribute will only store the attribute observed locally, however
        # we are interested in the global attribute, as the global attributes are 
        # used to separate the nodes from each other. Otherwise, one will get too many
        # syclic nodes in the resulting tree, as all local attributes will converge 
        # onto 0, 1, 2, ... 
        local_attribute = importance_func(current_data_node_list)
        global_attribute = available_attributes[local_attribute]
        available_attributes.remove(global_attribute) 

        root_node = nodes.DataNode(
          data=np.empty_like(current_data_node_list[0].get_data()),
          node_type=None,
          children=[],
          attribute=global_attribute 
        )

        vals = range(self.__min_val, self.__max_val + 1)

        for val in vals: 
          next_node_list = []
          for node in current_data_node_list:
            # Extract the nodes that match 
            node_data = node.get_data()
            if node_data.size == 0:
              warnings.warn("Node contains no data!")
              continue
            if node_data[local_attribute] == val:
              # Add to array for further invokations
              next_node_list.append(copy.copy(node))

          # Removing the attribute for all of the selected nodes
          for (idx, node) in enumerate(next_node_list):
            updated_data = np.delete(node.get_data(), local_attribute)
            next_node_list[idx].set_data(updated_data)

          # Next set of recursion
          root_node.add_child(
            child=learn_decision_tree(
              importance_func=importance_func,
              current_data_node_list=next_node_list,
              prev_data_node_list=current_data_node_list,
              available_attributes=copy.copy(available_attributes),
              tree_depth=tree_depth
            ),
            label=val
          ) 

        # If original root, save it
        # Otherwise return the root node
        if tree_depth == 1:
          self.__root_node = root_node
        return root_node
    
    learn_decision_tree(
      importance_func=importance_func,
      current_data_node_list=self.__training_nodes, 
      prev_data_node_list=[],
      available_attributes=copy.copy(self.__training_attributes_list)
    )


  def document_tree(
        self,
        root_node : nodes.DataNode  = None,
        comment   : str             = "Decision tree",
        id        : int             = 0,
        save_tree : bool            = True,
        show_tree : bool            = False
      ) -> None:

    def get_node_name(
          node : nodes.DataNode
        ) -> tuple:#[str, str]:
      data_type = node.get_type()
      attribute = node.get_attribute()
      # Check for invalid combination
      # If both the attribute not None and data_type not None, one cannot
      # know whether it is an internal node or a leaf-node
      if data_type is not None and attribute is not None:
        raise ValueError("Cannot document a tree with current_data_type and current_attribute both being defined")
      
      # Add node to tree
      if data_type is not None:
        # Leaf node. Should not contain children
        node_name = str(data_type)
        node_label = node_name
      else:
        # Internal node. Should contain children
        node_name = str(attribute)
        node_label = "A" + node_name
      return (node_name, node_label)


    def build_documented_tree(
        tree            : graphviz.Digraph, 
        current_node    : nodes.DataNode, 
        parent_node     : nodes.DataNode    = None,
        accounted_nodes : list              = [],
        label           : str               = ""
      ) -> graphviz.Digraph:

      # Create node
      (current_node_name, current_node_label) = get_node_name(node=current_node)
      tree.node(name=current_node_name, label=current_node_label)

      # Add edges to any potential parents
      if parent_node is not None:
        # Parent is an internal node
        (parent_node_name, _) = get_node_name(node=parent_node)

        tree.edge(tail_name=parent_node_name, head_name=current_node_name, label=label)

      # Iterate through all of the children, but only if they are not accounted for earlier
      # This is to prevent asyclic behaviour if a node has multiple parents
      for (child_node, val) in current_node.get_children():
        child_node_name = get_node_name(node=child_node)
        if child_node_name in accounted_nodes:
          continue
        
        accounted_nodes.append(child_node_name)
        build_documented_tree(
          tree=tree, 
          current_node=child_node, 
          parent_node=current_node,
          accounted_nodes=accounted_nodes, 
          label=str(val)
        )         

      return tree

    def save_tree(
        tree : graphviz.Digraph,
        show : bool             = False 
      ) -> None:
      tree.render(view=show)

    if root_node is None and self.__root_node is None:
      raise ValueError("No node found to document the tree")

    if root_node is None:
      root_node = self.__root_node

    # This method tries to document the tree using graphviz
    tree = graphviz.Digraph(comment=comment, filename=os.path.join(sys.path[0], "data/results/data/tree/{} id={}.gv".format(comment, id)))
    tree = build_documented_tree(
      tree=tree, 
      current_node=root_node, 
      parent_node=None,
      accounted_nodes=[]
    )
    if not save_tree:
      return
    save_tree(tree=tree, show=show_tree)

  def test_decision_tree(
        self
      ) -> None:
    """
    Testing how well the trained tree actually is.
    Returns a score of how accurate the algorithm has been into learning
    the system
    """

    def get_matching_leaf_node(
          root_node : nodes.DataNode,
          test_node : nodes.DataNode
        ) -> nodes.DataNode:
      # Tries to get the leaf-node that matches the test-node

      # Copy to prevent stupid python-errors that have plagued my project
      tree_node = copy.copy(root_node)
      node_data = test_node.get_data()

      while tree_node.get_attribute() is not None:
        attribute = tree_node.get_attribute()
        children = tree_node.get_children()

        found_value = False 

        # Find which child matches
        for (child, val) in children:
          if node_data[attribute] == val:
            found_value = True 
            tree_node = child
            break

        if not found_value:
          # No child supporting the test was found
          return None

      return copy.copy(tree_node)
      

    test_nodes = self.__testing_nodes
    num_correct = 0

    # Iterate through all of the nodes, and test if they are classified correctly
    for test_node in test_nodes:
      # Start from the root for all test-nodes
      # Get a leaf node 
      leaf_node = get_matching_leaf_node(root_node=self.__root_node, test_node=test_node)

      if leaf_node is None:
        # No matching leaf node found
        continue
      
      # Once a leaf-node is found, we must test if the detected value is correct
      tree_type = leaf_node.get_type()
      if test_node.get_type() == tree_type:
        num_correct += 1

    # print("Number correct {}, of {} possible. Proportion correct on test-set: {}".format(num_correct, len(test_nodes), num_correct / len(test_nodes)))
    return num_correct

  def extract_data(
        self,
        data_csv : TextIO,
      ) -> list:#[nodes.DataNode]:
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

  importance_class = importance.Importance(min_val=1, max_val=2)
  random_importance_func = lambda x : importance_class.random(x)
  expected_information_importance_func = lambda x : importance_class.expected_information(x)

  num_tests = 10 #5000
  random_correct_arr = np.zeros(num_tests, dtype=int)
  expected_correct_arr = np.zeros(num_tests, dtype=int)

  # Random tree
  random_tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )

  # Expected information tree
  expected_information_tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )

  # Iterating over the algorithm for 1000 iterations to create some data
  for i in range(num_tests):
    random_tree.train_decision_tree(importance_func=random_importance_func)
    random_tree.document_tree(
      root_node=None, 
      comment="Random importance",
      id=i,
      save_tree=False,
      show_tree=False
    )
    random_result = random_tree.test_decision_tree()

    expected_information_tree.train_decision_tree(importance_func=expected_information_importance_func)
    expected_information_tree.document_tree(
      root_node=None, 
      comment="Expected information importance",
      id=i,
      save_tree=False,
      show_tree=False
    )
    expected_result = expected_information_tree.test_decision_tree()

    # Save data temporary
    random_correct_arr[i] = int(random_result)
    expected_correct_arr[i] = int(expected_result)

    if i % 50 == 0:
      print(i / num_tests)

  # Save data permanently
  np.savetxt(os.path.join(sys.path[0], "data/results/data/random_results.txt"), random_correct_arr)
  np.savetxt(os.path.join(sys.path[0], "data/results/data/expected_results.txt"), expected_correct_arr)

  # Calculate some statistics
  random_correct_mean = np.mean(random_correct_arr)
  random_correct_var = np.var(random_correct_arr)

  expected_correct_mean = np.mean(expected_correct_arr)
  expected_correct_var = np.var(expected_correct_arr)

  # Find percentile of times where the expected is better than the random
  num_random_similar = len(random_correct_arr[random_correct_arr == expected_correct_mean])
  num_random_better = len(random_correct_arr[random_correct_arr > expected_correct_mean])

  # Plot histograms
  fig, axs = plt.subplots(2)
  fig.suptitle(
    "Histogram comparison for {} tests.\n Random importance function similar in {} test(s) and better in {} test(s)".format(
      num_tests,
      num_random_similar,
      num_random_better
    )
  )

  axs[0].set_title(
    "Random importance function with mean: {} and variance: {}".format(
      '%.1f'%random_correct_mean, 
      '%.1f'%random_correct_var
    )
  )
  axs[0].hist(random_correct_arr, bins=np.arange(12,29,1))

  axs[1].set_title(
    "Expected information importance function with mean: {} and variance: {}".format(
      '%.1f'%expected_correct_mean, 
      '%.1f'%expected_correct_var
    )
  )
  axs[1].hist(expected_correct_arr, bins=np.arange(12,29,1))

  plt.show()


