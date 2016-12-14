import numpy as np
import sklearn.cluster
import copy
from sklearn import mixture
#from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
#from joblib import Parallel, delayed
from sklearn.externals.joblib import Parallel, delayed
from sklearn.base import clone #clones a classifier, but not the data
from scipy.sparse import csc_matrix
from sklearn.decomposition import *


# This method finds the codebook given a matrix 'data' that follows the next
# structure: n_samples x n_features
# Returns the codebook and (optionally) the KMeans estimator
# Some clustering methods are allowed: gmm, dpgmm, kmeans, and random (see
# theirs implementations at http://scikit-learn.org/stable/modules/clustering)
# TODO: For the moment, the allowed distance function is the euclidean
# distance, however I intend to create my own k-means or use other function
# (www.spectralpython.net/class_func_ref.html?highlight=kmeans#spectral.kmeans)
def build_codebook(data_train, number_of_words=300, clustering_method='kmeans',
  distance_function='euclidean', random_seed = 0, **_):
  if clustering_method == 'kmeans':
    KMeans = sklearn.cluster.KMeans(
        init = 'k-means++',
        n_clusters=number_of_words,
        n_jobs=-1, tol=1e-6,
        max_iter=1000
        ).fit(data_train)
    words = KMeans.cluster_centers_
    return (words, KMeans)
  elif clustering_method == 'gmm':
    gmm = mixture.GMM(n_components = number_of_words, n_iter=1000, tol=1e-6)
    gmm.fit(data_train)
    words = gmm.means_
    return (words, gmm)
  elif clustering_method == 'dpgmm':
    gmm = mixture.DPGMM(n_components = number_of_words)
    gmm.fit (data_train)
    words = gmm.means_
    return (words, gmm)
  elif clustering_method == 'random':
    np.random.seed(random_seed)
    idx = range(data_train.shape[0])
    np.random.shuffle (idx)
    words = data_train[idx[:number_of_words], :]
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(words, range(number_of_words))
    return (words, knn)


# Returns the feature histogram for a single video
# code_histogram contains the results of coding_per_video, it has the form:
#  n_samples x n_codewords
# The method supports: max, min, sum, mean pooling types
def pooling_per_video(code_histograms, type_pooling='max'):
  histograms_class = code_histograms.__class__
  if type_pooling == 'sum':
    pooling_function = histograms_class.sum
  elif type_pooling == 'min':
    pooling_function = histograms_class.min
  elif type_pooling == 'max':
    pooling_function = histograms_class.max
  elif type_pooling == 'mean':
    pooling_function = histograms_class.mean

  descriptor = pooling_function(code_histograms, axis=0)

  # check if it is a csc_matrix and type_pooling == (max or min)
  if histograms_class.__name__[0] == 'c' and type_pooling[2] in ('n', 'x'):
    return descriptor.toarray()[0]
  return np.array(descriptor)[0]


# This funciton is only used in coding_per_video for parallel purposes
def call_predictor(predictor, arguments):
  return predictor.predict(arguments)


# Returns the histogram of codewords given a matrix data
# (n_samples x n_features) and the codebook (which actually it is not used,
# instead the KMeans parameter should be passed.
# This methods implements hard pooling, soft pooling does not work yet.
# TODO: check this function since KMeans can carry a lot of unuseful information
# TODO: make it parallel. However, we can use the coding_pooling in parallel
# when calling the function coding_pooling_per_video
def coding_per_video(codebook_predictor, number_of_words, data_per_video, type_coding='hard'):
  hist = None
  if type_coding == 'hard':
    col = [(codebook_predictor.predict (data_point_features.reshape(1, -1)))[0] for data_point_features in data_per_video]
    hist = csc_matrix(
            (np.ones(data_per_video.shape[0]), (np.arange(data_per_video.shape[0]), col)),
            shape=(data_per_video.shape[0], number_of_words),
            dtype=np.float)
  elif type_coding == 'soft':
    return None
    # TODO: implement soft pooling
  return hist


# Runs the cooding and pooling
def coding_pooling_per_video (codebook_predictor, number_of_words, data_per_video, type_coding='hard', type_pooling='sum', **_):
  hist = coding_per_video (codebook_predictor = codebook_predictor, number_of_words = number_of_words, data_per_video = data_per_video, type_coding=type_coding);
  return pooling_per_video (code_histograms = hist, type_pooling = type_pooling);


# Saves the codebook to a file in the directory
def save_codebook (KMeans, file_to_save):
  file_to_open += '' if len (file_to_open) and file_to_open[-4:] == '.pkl' else '.pkl';
  joblib.dump (KMeans, file_to_save);


# Loads the codebook from the file
def load_codebook (file_to_open):
  file_to_open += '' if len (file_to_open) and file_to_open[-4:] == '.pkl' else '.pkl';
  return joblib.load (file_to_open);


# Sparse Coding
# TODO: check sparse coding
def train_sparse_codes (data, n_components, alpha=1, max_iter=1000, tol=1e-08, fit_algorithm='lars', transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, code_init=None, dict_init=None, verbose=False, split_sign=False, random_state=None, batch_size=3, shuffle=True, use_minibatch = True, n_processors = 1):
  if use_minibatch:
    return MiniBatchDictionaryLearning (n_components = n_components, alpha=alpha, n_iter=max_iter, fit_algorithm=fit_algorithm, batch_size=batch_size, shuffle=shuffle, dict_init=dict_init, transform_algorithm=transform_algorithm, transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_alpha=transform_alpha, verbose=verbose, split_sign=split_sign, random_state=random_state, n_jobs = n_processors).fit (data)
  else:
    return DictionaryLearning (n_components = n_components, alpha=alpha, max_iter=max_iter, tol=tol, fit_algorithm=fit_algorithm, transform_algorithm=transform_algorithm, transform_n_nonzero_coefs=transform_n_nonzero_coefs, transform_alpha=transform_alpha, code_init=code_init, dict_init=dict_init, verbose=verbose, split_sign=split_sign, random_state=random_state, n_jobs = n_processors).fit (data)


def coding_pooling_SC (sparsecode_model, data, type_pooling = 'sum'):
  hist = sparsecode_model.transform (data)
  return pooling_per_video (code_histograms = hist, type_pooling = type_pooling)
