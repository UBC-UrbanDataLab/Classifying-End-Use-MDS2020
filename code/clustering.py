#!/usr/bin/env python
# coding: utf-8

### ~ Library Imports ~ ###
# Data storing Imports
import numpy as np

# Clustering imports
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import hdbscan
import skfuzzy as fuzz

# Cluster performance comparison metric imports
from sklearn.metrics import silhouette_score

# Distance Calculation Imports
import gower

# MDS Import
from sklearn.manifold import MDS

### ~ Calculating the Gower's distances ~ ###
def make_categorical_list(cont_idxs, num_predictors):
    """Function to generate a list of booleans to identify which columns are categorical and which aren't
    Args:
        cont_idxs (list): list of indexes that are continuous
        num_predictors (int): number of predictor variables
        
    Returns:
        is_cat (list): a list of boolean values indicating which columns are categorical (True) and which are continuous (False)
    """
    if type(cont_idxs)==int: cont_idxs = [cont_idxs]
    is_cat = []
    for i in range(num_predictors):
        if i in cont_idxs:
            is_cat.append(False)
        else:
            is_cat.append(True)
    return is_cat

def calc_gowers(df, continuous_columns):
    """Function to simplify calculating Gower's distance
    Args:
        df (pandas.DataFrame): dataframe of observations to calculate Gower's dstance for
        continuous_columns (list): list of integers identifying the indexes of columns that are continuous
        
    Returns:
        gow_dists (numpy.array): a numpy array of Gower's distances between observations
    """
    catList = make_categorical_list(continuous_columns, len(df.columns)-5)
    data_np = df.iloc[:,5:].to_numpy()
    gow_dists = gower.gower_matrix(data_np, cat_features=catList)
    return gow_dists

### ~ Multidimensional Scaling ~ ###
def multidim_scale(dist_mat, num_dim=2):
    """Function to perform Multidimensional Scaling (MDS)
    Args:
        dist_mat (numpy.array): a numpy array of relative distances betwen observations
        num_dim (int): the number of dimensions to scale down to (default=2)
        
    Returns:
        scaled_data (numpy.array): a numpy array of values that satisfy the distance measures passed in
    """
    scaler = MDS(n_components=num_dim, dissimilarity='precomputed')
    scaled_data = scaler.fit_transform(dist_mat)
    return scaled_data

### ~ Clustering Function ~ ###
def cluster(df, clust_type, num_clusts = None, continuous_columns = None, input_type='original'):
    """Function to simplify various different clustering methods
    
       Available model types are: kmeans, agglom (agglomerative or hierarchical), dbscan, hdbscan, 
       gmm (gausian mixture model), vbgm (variational bayesian gaussian mixture model), meanshift, 
       and fuzzy (fuzzy c-means)
    Args:
        df (pandas.DataFrame): dataframe of original values (if input_type='original')
        OR                     numpy array of Gower's distances between observations (if input_type='gowers')
        OR                     dataframe of multidimensionaly scaled values (if input_type='mds')
        clust_type (str): the type of clustering to be done options are kmeans, agglom, dbscan, hdbscan, gmm, 
                          vbgm, meanshift, and fuzzy
        num_clusts (int): the number of clusters to generate (default=None) only required for kmeans, agglom, 
                          gmm, vbgm, fuzzy
        continuous_columns (list): list of integers identifying the indexes of columns that are continuous
        input_type (str): user defined input type options are 'original', 'gowers', and 'mds'
        
    Returns:
        preds (numpy.array): a numpy array of the cluster groups
        OR
        None: if clustering doesn't work and/or the sihlouette score can't be calculated
    """
    needs_mds = ['kmeans', 'gmm', 'vbgm', 'meanshift', 'fuzzy'] # List of models that don't accept distance measures as input
    # Determines how to address the input (calculate gower's distance, store, or use as is) based on the user defined input type
    ## Allows the user to pass in either data on the original scale, as gowers distance or after performing MDS 
    ## Allows for calculating Gowers distance and MDS once before calling this function and passing it in rather than recalculating every time
    if input_type == 'original':
        fit_data = calc_gowers(df, continuous_columns)
    elif input_type == 'gowers':
        fit_data = df
    elif input_type == 'mds':
        fit_data = df
        needs_mds = []
   # Performs MDS on the Gower's distances (if needed) on the model types that require coordinate values as inputs
   # (this process allows for clustering mixed continuous and categorical within the same predictor space)     
    if clust_type in needs_mds:
        fit_data = multidim_scale(fit_data, num_dim=3) # len(df.iloc[0,:])-8 can be used to make num_dim=# of predictor variables (slight silhouette score improvement)
    # Runs the user defined clustering method
    if clust_type == 'kmeans':
        # Needs df, clust_type, num_clusts, continuous_columns
        model = KMeans(n_clusters=num_clusts).fit(fit_data)
        preds = model.labels_
    elif clust_type == 'agglom':
        # Needs df, clust_type, num_clusts, continuous_columns
        if input_type != 'mds':
            model = AgglomerativeClustering(affinity='precomputed', linkage = 'single', n_clusters=num_clusts).fit(fit_data)
        else:
            model = AgglomerativeClustering(linkage = 'single', n_clusters=num_clusts).fit(fit_data)
        preds = model.labels_
    elif clust_type == 'dbscan':
        # Needs df, clust_type, continuous_columns
        if input_type == 'mds':
            print("DBSCAN doesn't accept mds input type, please provide input as original unscaled data or as a Gower's distance matrix.")
        else:
            model = DBSCAN(eps=0.15, min_samples=5, metric='precomputed').fit(fit_data)
            preds = model.labels_
    elif clust_type == 'hdbscan':
        # Needs df, clust_type, continuous_columns
        if input_type == 'mds':
            print("HDBSCAN doesn't accept mds input type, please provide input as original unscaled data or as a Gower's distance matrix.")
        else:
            model = hdbscan.HDBSCAN(metric='precomputed', cluster_selection_epsilon=0.15).fit(fit_data.astype('double'))
            preds = model.labels_
    elif clust_type == 'gmm':
        # Needs df, clust_type, num_clusts, continuous_columns
        model = GaussianMixture(n_components=num_clusts, covariance_type='spherical',n_init=100).fit(fit_data)
        preds = model.predict(fit_data)
    elif clust_type == 'vbgm':
        # Needs df, clust_type, num_clusts, continuous_columns
        model = BayesianGaussianMixture(n_components=num_clusts, covariance_type='spherical',n_init=100).fit(fit_data)
        preds = model.predict(fit_data)
    elif clust_type == 'meanshift':
        # Needs df, clust_type, continuous_columns
        model = MeanShift(cluster_all=False).fit(fit_data) # cluster_all=False means that items that don't belong to a cluster or not included in any cluster (assigned to -1)
        preds = model.predict(fit_data)
    elif clust_type == 'fuzzy':
        # Needs df, clust_type, num_clusts, continuous_columns
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(np.transpose(fit_data), num_clusts, 4, error=0.005, maxiter=1000, init=None)
        preds = np.argmax(u, axis=0)
    else:
        print("Specified model type not available yet")
        return None
    try:
        # Silhouette Score appears to be the prefered cluster comparison metric used with sklearn from what I have read (1 is good, -1 is bad)
        print("Silhouette Score for {} clustering: {}".format(clust_type, silhouette_score(fit_data, preds)))
        return preds
    except:
        print('Unable to calculate silhouette score for the given model under the current conditions')
        return None