import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import heapq
from scipy.spatial import distance
from tqdm.notebook import tqdm, trange
import copy
import heapq
from scipy import spatial
import os


# get only indexes of nearest neighbors
def get_neighbor_indexes(knns):
  indexes = []
  for n in knns:
    indexes.append(n[1])

  return indexes

#from IPython.core.display import display_javascript
class LocalityHash:

  def __init__(self, hash_size, input_dim, num_tables = 1):
      
      """
        arg input_dim: the expected dimension of the data to be hashed
        arg hash_size: number of random hyperplanes that divide the space into hash buckets
        
        Info: What we are realy storing for hyperplanes is the norm of the planes, initialized randomly with standard normal distribution. 
        Points whose dot product with the norm have the same sign are on the same side of the plane.
      """
      self.hash_size = hash_size
      self.input_dim = input_dim
      self.num_tables = num_tables

      self.planes = [np.random.randn(self.hash_size, self.input_dim) for _ in range(self.num_tables)]
      self.hash_tables = [{} for _ in range(self.num_tables)]

      print("LSH initialized")
      print("Number of hash tables: {}".format(self.num_tables))
      print("Number of hyperplanes per table: {}".format(self.hash_size))
      print("Number of buckets per table: {}".format(2**(self.hash_size)))
  
  def get_hashkey(self, plane, input_point):

    """ 
      arg input point dimension = 1* input_dim

      Takes an input point and generates a hash key based on its location relative to the hyperplanes (In which bucket)
      Returns a binary string map of length self.hash_size (number of hyperplanes) where 1 = "above" hyperplane i and 0 = "below" hyperplane i
      looks like: "101010100"

    """
    location_projections = np.dot(plane, input_point)
        
    # if dot product of input point and hyper plane is positive, its 'above' in n-d space, if negative or 0, below
    hash_key = "".join(['1' if i > 0 else '0' for i in location_projections])

    return hash_key


  def insert_point(self, input_point, original_index):

    """
      arg input_point dimension = 1* input_dim
      arg original_index = reference to the index of the full size image for retrieval
    """

    info = (tuple(input_point.tolist()), original_index)  #save point as tuple to save memory
    
    for i, hash_table in enumerate(self.hash_tables):
      key = self.get_hashkey(self.planes[i], input_point)

      if key in hash_table:
          hash_table[key].append(info)
      else:
          hash_table[key] = [info]

  
  def initialize(self, np_data, verbose = False):
    if not isinstance(np_data[0], np.ndarray) or not np_data.shape[1] == self.input_dim:
        print("Input should be an array of np array vectors of dimension: 1 x {}".format(self.input_dim))
        return

    print("Initializing LSH from data...")
    print("Input data shape: {}".format(np_data.shape))

    for i, point in enumerate(np_data):
      self.insert_point(point, i)

    if verbose:
      count_str = "\nInserted:\n"
      for i, hash_table in enumerate(self.hash_tables):
        for bucket, item in hash_table.items():
            count_str += "{} items into bucket {} in hash table {}".format(str(len(item)), bucket, str(i+1))
            count_str += "\n"
        count_str += "\n"

      print(count_str)



  def get_knns(self, query_point, k):
      
      candidates = set()

      for i, table in enumerate(self.hash_tables):
          key = self.get_hashkey(self.planes[i], query_point)
          if key in table:
            candidates.update(table[key])

      if len(candidates) <= 1:  #if query returns nothing or just the query point itself, return
        if len(candidates) == 0 or (tuple(candidates)[0][0]== tuple(query_point.tolist())):
          print("No neighbors found")
          return
    
      #add eucledian:
      candidates_aug = []
      for i, (point, index) in enumerate(candidates):
          dist = np.linalg.norm(point - query_point)
          candidates_aug.append((point, index, dist))
      
      #sort on eucledian
      candidates_aug.sort(key=lambda x: x[2])

      ##trim if duplicates/remove same value:
      smallest_dist = candidates_aug[0][2]
      i = 0
      while i < len(candidates_aug) and smallest_dist == 0:
        candidates_aug = candidates_aug[i+1:]
        smallest_dist = candidates_aug[0][2]

      if len(candidates_aug) <= k:
        return candidates_aug
      else:
        return candidates_aug[:k]

