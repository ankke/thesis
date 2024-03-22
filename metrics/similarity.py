"""
Module implementing various feature representation similarity metrics.
Adapted from https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
"""

import gc
from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from matplotlib import pyplot as plt
import numpy as np
from ignite.exceptions import NotComputableError
import torch


def gram_linear(x):
  """Compute Gram (kernel) matrix for a linear kernel.

  Args:
    x: A num_examples x num_features matrix of features.

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
  """Compute Gram (kernel) matrix for an RBF kernel.

  Args:
    x: A num_examples x num_features matrix of features.
    threshold: Fraction of median Euclidean distance to use as RBF kernel
      bandwidth. (This is the heuristic we use in the paper. There are other
      possible ways to set the bandwidth; we didn't try them.)

  Returns:
    A num_examples x num_examples Gram matrix of examples.
  """
  dot_products = x.dot(x.T)
  sq_norms = np.diag(dot_products)
  sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
  sq_median_distance = np.median(sq_distances)
  return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
  """Center a symmetric Gram matrix.

  This is equvialent to centering the (possibly infinite-dimensional) features
  induced by the kernel before computing the Gram matrix.

  Args:
    gram: A num_examples x num_examples symmetric matrix.
    unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
      estimate of HSIC. Note that this estimator may be negative.

  Returns:
    A symmetric matrix with centered columns and rows.
  """
  gram = gram.copy()

  if unbiased:
    # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
    # L. (2014). Partial distance correlation with methods for dissimilarities.
    # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
    # stable than the alternative from Song et al. (2007).
    n = gram.shape[0]
    np.fill_diagonal(gram, 0)
    means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
    means -= np.sum(means) / (2 * (n - 1))
    gram -= means[:, None]
    gram -= means[None, :]
    np.fill_diagonal(gram, 0)
  else:
    means = np.mean(gram, 0, dtype=np.float64)
    means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]

  return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
    gram_x: A num_examples x num_examples Gram matrix.
    gram_y: A num_examples x num_examples Gram matrix.
    debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
    The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

def upsample_examples(X,Y):
    """
    Upsamples the set (of X or Y) with less samples such that both sets contain the same number of samples"""
    if X.shape[0] < Y.shape[0]:
        X = np.repeat(X, Y.shape[0] // X.shape[0], axis=0)
        Y = Y[:X.shape[0]]
    elif X.shape[0] > Y.shape[0]:
        Y = np.repeat(Y, X.shape[0] // Y.shape[0], axis=0)
        X = X[:Y.shape[0]]

    return X, Y

def downsample_examples(X,Y):
    """
    Downsamples the set (of X or Y) with more samples such that both sets contain the same number of samples"""
    if X.shape[0] < Y.shape[0]:
        Y = Y[:X.shape[0]]
    elif X.shape[0] > Y.shape[0]:
        X = X[:Y.shape[0]]

    return X, Y


def batch_cka(X, Y):
    """
    Computes the cka for a complete batch of samples by averaging the pairwise distance between each example from X with each example from Y.
    Args:
        X: A num_examples_source x num_features matrix of features.
        Y: A num_examples_target x num_features matrix of features.
    """
    gram_x = gram_rbf(X, threshold=0.5)
    gram_y = gram_rbf(Y, threshold=0.5)
    res = cka(gram_x, gram_y)
    # Takes absolute value of res
    res = np.abs(res)
    return res

def batch_cosine(X, Y):
    """
    Computes the cosine similarity for a complete batch of samples by averaging the pairwise distance between each example from X with each example from Y.
    Args:
        X: A num_examples_source x num_features matrix of features.
        Y: A num_examples_target x num_features matrix of features.
    """
    X,Y = batch_cross_product(X, Y)

    # Computing cosine similarity
    res = np.sum(X * Y, axis=1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y, axis=1))

    return np.mean(np.abs(res))

def batch_euclidean(X, Y):
    """
    Computes the euclidean distance for a complete batch of samples by averaging the pairwise distance between each example from X with each example from Y.
    Args:
        X: A num_examples_source x num_features matrix of features.
        Y: A num_examples_target x num_features matrix of features.
    """
    X,Y = batch_cross_product(X, Y)

    # Computing euclidean distance
    res = np.linalg.norm(X - Y, axis=1)

    return np.mean(np.abs(res))

def batch_cross_product(X, Y):
    """
    Building pairs such that each example from X is paired with each example from Y
    """
    X = np.repeat(X, Y.shape[0], axis=0)
    Y = np.tile(Y, (X.shape[0] // Y.shape[0], 1))

    return X, Y
  
class SimilarityMetricPCA:
  """
  Accumulates a similarity metric over multiple iterations.
  """

  def __init__(self, device="cpu", similarity_function=batch_cka, base_metric=None, max_datapoints=4*10000):
    self._source_batches = []
    self._target_batches = []
    self.X, self.Y = None, None
    self.computed = False
    self._similarity_function = similarity_function
    self.device = device
    self.max_datapoints = max_datapoints
    self.num_source_datapoints = 0
    self.num_target_datapoints = 0
    if base_metric is not None:
       self.base_metric = base_metric
    else:
       self.base_metric = None
    
  def reset(self):
    self.computed = False
    if not self.base_metric:
      del self._source_batches
      del self._target_batches
      gc.collect()
      torch.cuda.empty_cache()

      self._source_batches = []
      self._target_batches = []
      self.num_source_datapoints = 0
      self.num_target_datapoints = 0

  def update(self, features, domains):
    self.computed = False

    if self.base_metric is None:
      if self.num_source_datapoints < self.max_datapoints:
        X = features[domains == 0].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu()
        self._source_batches.append(X)
        self.num_source_datapoints += X.shape[0]
      if self.num_target_datapoints < self.max_datapoints:
        Y = features[domains == 1].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu()
        self._target_batches.append(Y)
        self.num_target_datapoints += Y.shape[0]
        
    del features
    del domains


  def get_features(self):
    if self.computed: 
      return self.X, self.Y
    else:
      # Concatenate list of to single torch tensors
      X = torch.cat(self._source_batches, axis=0)
      Y = torch.cat(self._target_batches, axis=0)

      X, Y = downsample_examples(X, Y)
      X = torch.pca_lowrank(X, q=50)[0].numpy()
      Y = torch.pca_lowrank(Y, q=50)[0].numpy()
      self.X, self.Y = X, Y

      del self._source_batches
      del self._target_batches
      self._source_batches = []
      self._target_batches = []
      self.num_source_datapoints = 0
      self.num_target_datapoints = 0
      gc.collect()

      self.computed = True
    return self.X, self.Y

  def compute(self):
    if self.base_metric is not None:
      X, Y = self.base_metric.get_features()
    else:
      X, Y = self.get_features()

    try:
      similarity = self._similarity_function(X,Y)
    except:
      similarity = np.NaN

    return similarity