#!/usr/bin/env python
# coding: utf-8

### ~ Library Imports ~ ###
# Data storing Imports
import numpy as np
import pandas as pd

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

# Function imports from other files
import data_preparation as dp

# Plotting Imports
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

### ~ Calculating the Gower's distances ~ ###
# Generating list of Booleans for identifying which columns are categorical and which aren't (pass in a list of columns that are continuous and the number predictor variables)
def make_categorical_list(cont_idxs, num_predictors):
    if type(cont_idxs)==int: cont_idxs = [cont_idxs]
    is_Cat = []
    for i in range(num_predictors):
        if i in cont_idxs:
            is_Cat.append(False)
        else:
            is_Cat.append(True)
    return is_Cat

# Function to simplify calculating Gower's distance (pass in the dataframe and the indexes of the columns that are categorical)
def calc_gowers(df, continuous_columns):
    catList = make_categorical_list(continuous_columns, len(df.columns)-8)
    data_np = df.iloc[:,8:].to_numpy()
    gow_dists = gower.gower_matrix(data_np, cat_features=catList)
    return gow_dists

### ~ Multidimensional Scaling ~ ###
# Function to perform Multidimensional Scaling (MDS)
def multidim_scale(dist_mat, num_dim=2):
    scaler = MDS(n_components=num_dim, dissimilarity='precomputed')
    scaled_data = scaler.fit_transform(dist_mat)
    return scaled_data

### ~ Clustering Function ~ ###
# Function to quickly run various different clustering methods, look at the portion of the if statement to see what values each model type needs in order to run
## Available model types are: kmeans, agglom (agglomerative or hierarchical), dbscan, hdbscan, gmm (gausian mixture model), 
##                            vbgm (variational bayesian gaussian mixture model), meanshift, and fuzzy (fuzzy c-means)
def cluster(df, clust_type, num_clusts = None, continuous_columns = None, input_type='original'):
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
            model = AgglomerativeClustering(affinity='precomputed', linkage = 'complete', n_clusters=num_clusts).fit(fit_data)
        else:
            model = AgglomerativeClustering(linkage = 'complete', n_clusters=num_clusts).fit(fit_data)
        preds = model.labels_
    elif clust_type == 'dbscan':
        # Needs df, clust_type, continuous_columns
        if input_type == 'mds':
            print("DBSCAN doesn't accept mds input type, please provide input as original unscaled data or as a Gower's distance matrix.")
        else:
            model = DBSCAN(eps=0.06, min_samples=2, metric='precomputed').fit(fit_data)
            preds = model.labels_
    elif clust_type == 'hdbscan':
        # Needs df, clust_type, continuous_columns
        if input_type == 'mds':
            print("HDBSCAN doesn't accept mds input type, please provide input as original unscaled data or as a Gower's distance matrix.")
        else:
            model = hdbscan.HDBSCAN(metric='precomputed', cluster_selection_epsilon=0.06).fit(fit_data.astype('double'))
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
    try:
        # Silhouette Score appears to be the prefered cluster comparison metric used with sklearn from what I have read (1 is good, -1 is bad)
        print("Silhouette Score for {} clustering: {}".format(clust_type, silhouette_score(fit_data, preds)))
        return preds
    except:
        print('Unable to calculate silhouette score for the given model under the current conditions')