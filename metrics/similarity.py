"""
Module implementing the CKA metric on a batch level.
Adapted from https://colab.research.google.com/github/google-research/google-research/blob/master/representation_similarity/Demo.ipynb#scrollTo=MkucRi3yn7UJ
"""

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import numpy as np
from ignite.exceptions import NotComputableError
import torch

from metrics.svcca import robust_cca_similarity

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

class SimilarityMetric(Metric):
  """
  Accumulates a similarity metric over multiple iterations."""

  def __init__(self, output_transform=lambda x:x, device="cpu"):
    self._source_samples = []
    self._target_samples = []
    super(SimilarityMetric, self).__init__(output_transform=output_transform, device=device)

  @reinit__is_reduced
  def reset(self):
    self._source_samples = []
    self._target_samples = []
    super(SimilarityMetric, self).reset()

  @reinit__is_reduced
  def update(self, output):
    features, domains = output
    X = features[-1][domains == 0].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu().numpy()
    Y = features[-1][domains == 1].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu().numpy()
    self._source_samples.append(X)
    self._target_samples.append(Y)


  @sync_all_reduce("_num_examples", "_num_correct:SUM")
  def compute(self):
    if self._num_examples == 0:
        raise NotComputableError('CustomAccuracy must have at least one example before it can be computed.')
    
    # Concat list of np arrays across first dimension
    X = np.concatenate(self._source_samples, axis=0)
    Y = np.concatenate(self._target_samples, axis=0)

    return batch_cka(X, Y)