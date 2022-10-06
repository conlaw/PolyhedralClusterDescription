import pandas as pd
import math
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


def extractW(res, W, b):
    #Pull out selected halfspaces
    z = res['z']
    W_final = {}
    B_final = {}
    starting_idx = 0
        
    for cl in W.keys():
        W_final[cl] = {}
        B_final[cl] = {}
        num_hps = len(W[cl].keys())

        for key in np.array(list(W[cl].keys()))[np.array(z[starting_idx:(starting_idx+num_hps)]).astype(bool)]:
            W_final[cl][key] = W[cl][key]
            B_final[cl][key] = b[cl][key]
            
        starting_idx = starting_idx + num_hps
        
    return W_final, B_final
    

def compressData(data, cluster, alpha_type='global', estimator='dt', num_hps = -1):
    '''
    Helper to compress data and get starting set of axis-parallel hyperplanes
    '''
    #Get upper bound of prediction errors
    clf = tree.DecisionTreeClassifier(max_leaf_nodes = len(np.unique(cluster)))
    clf.fit(data,cluster)
    pred_errors = data.assign(pred_error = clf.predict(data) != cluster, cluster = cluster).groupby('cluster')['pred_error'].sum()
    
    
    #Compress Data + Generate starting halfspaces
    full_indices_to_keep = []
    
    if alpha_type == 'global':
        alpha = int(pred_errors.sum())
    elif alpha_type == 'local':
        alpha_dict = {}
    
    W = {}
    b = {}
    init_counter = 0
    
    if num_hps < 0:
        num_hps = alpha
    else:
        num_hps = num_hps
                                      
    for cl in np.unique(cluster):
        cl0 = data[cluster == cl]
        indices_to_keep = []
        W[cl] = {}
        b[cl] = {}
                                      
        if alpha_type == 'local':
            alpha = int(pred_errors.loc[cl])
            alpha_dict[cl] = alpha*1.2
                                      
        for id, column in enumerate(cl0.columns):
            w_base = np.zeros(data.shape[1])
            w_base[id] = 1                          
            if column == 'cluster':
                continue

            sorted_vals = cl0.sort_values(column)
            
            for val in np.unique(sorted_vals[column].head(alpha+1)):
                W[cl]['INIT_'+str(init_counter)] = w_base*-1
                b[cl]['INIT_'+str(init_counter)] = val
                init_counter += 1      
                                      
            for val in np.unique(sorted_vals[column].tail(alpha+1)):
                W[cl]['INIT_'+str(init_counter)] = w_base
                b[cl]['INIT_'+str(init_counter)] = val
                init_counter += 1                      

            indices_to_keep.append(sorted_vals.head(num_hps+1).index.values)
            indices_to_keep.append(sorted_vals.tail(num_hps+1).index.values)

        unique_indices = np.unique(np.array(indices_to_keep).reshape(-1))

        full_indices_to_keep = np.concatenate([full_indices_to_keep, np.unique(np.array(indices_to_keep).reshape(-1))])
                                      
    full_indices_to_keep = np.unique(full_indices_to_keep)
    
    if alpha_type == 'local':
        alpha = alpha_dict
    return data.loc[full_indices_to_keep], cluster.loc[full_indices_to_keep], W, b, alpha


def construct_meta_cluster(data, cluster, k=100, eps=0.5, method='hclust'):
    '''
    Helper to construct meta-clusters
    '''
    
    cl_data = data[data.cluster == cluster]
    
    if method == 'hclust':
        db = AgglomerativeClustering(n_clusters = None, linkage='complete',
                                     distance_threshold=eps).fit(cl_data.drop('cluster',axis=1))
        return cl_data.assign(group = db.labels_).reset_index(drop=True)
    
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=52, n_init=100).fit(cl_data.drop('cluster',axis=1))
        return cl_data.assign(group = kmeans.labels_).reset_index(drop=True)
    
    elif method == 'dbscan':
        db = DBSCAN(eps=eps).fit(cl_data.drop('cluster',axis=1))
        return cl_data.assign(group = db.labels_).reset_index(drop=True)
    
    else:
        print('No matching meta clustering method.')
        return
    
def computeActualErrors(feature_mat, cluster, W, B):
    '''
    Helper function to compute errors for given clustering + halfspaces
    '''
    error_mat = np.zeros((feature_mat.shape[0],len(W.keys())))

    for id2,cl in enumerate(W.keys()):
        num_const = len(W[cl].keys())
        res_mat = np.zeros((feature_mat.shape[0],num_const))
        for id,key in enumerate(W[cl].keys()):
            res_mat[:,id] = W[cl][key] @ feature_mat.T + B[cl][key] <= 0

        cl_membership = res_mat.all(axis=1)

        error_mat[:,id2] = (cluster == cl) != cl_membership

    total_errors = error_mat.any(axis=1)
    
    return total_errors.sum(), total_errors

def computeDistance(grouped_mat, W, B):
    '''
    Helper function to compute errors for given clustering + halfspaces
    '''

    dist_mat = np.zeros((grouped_mat.shape[0], len(list(W.keys()))))
    
    for id, cl in enumerate(list(W.keys())):
        cl_mat = np.zeros((grouped_mat.shape[0], len(list(W[cl].keys()))))
        for id2, w in enumerate(list(W[cl].keys())):
            w_concat = np.concatenate([-np.clip(-1*np.array(W[cl][w]).T.reshape(-1),0,None),
                                       np.clip(np.array(W[cl][w]).T.reshape(-1),0,None)])

            cl_mat[:,id2] = np.clip(np.abs(grouped_mat @ w_concat + B[cl][w]), 0, None)
        dist_mat[:,id] = cl_mat.min(axis=1)
    
    return dist_mat.min(axis=1)


#########
## Deprecated functions for breaking groups
##
#########
def break_group_eps(df, grouped_df, cl, group, eps = 0.05):
    db = DBSCAN(eps=eps).fit(df[(df.cluster == cl) & (df.group == group)].drop(['cluster','group'],axis=1))
    new_groups = df[(df.cluster == cl) & (df.group == group)].assign(group = lambda df: [str(group)+'-'+str(id) for id in db.labels_])
    
    df = (df[(df.cluster != cl) | (df.group != group)]
          .append(new_groups)
    )
    
    new_groups = (new_groups.groupby(['cluster','group']).max().reset_index()
                 .merge(new_groups.groupby(['cluster','group']).min().reset_index(), on=['cluster','group'],suffixes= ('_upper','_lower'))
                 .merge(new_groups.groupby(['cluster','group'])['0'].count().reset_index(),on=['cluster','group'])
                 .rename({'0':'weight'},axis=1)
                )
    return df, (grouped_df[(grouped_df.cluster != cl) | (grouped_df.group != group)]
            .append(new_groups)
           )

def break_group_k(df, grouped_df, cl, group, k = 3):
    kmeans= KMeans(n_clusters=k, random_state=42, n_init=1).fit(df[(df.cluster == cl) & (df.group == group)].drop(['cluster','group'],axis=1))
    new_groups = df[(df.cluster == cl) & (df.group == group)].assign(group = lambda df: [str(group)+'-'+str(id) for id in kmeans.labels_])
    
    df = (df[(df.cluster != cl) | (df.group != group)]
          .append(new_groups)
    )
    
    new_groups = (new_groups.groupby(['cluster','group']).max().reset_index()
                 .merge(new_groups.groupby(['cluster','group']).min().reset_index(), on=['cluster','group'],suffixes= ('_upper','_lower'))
                 .merge(new_groups.groupby(['cluster','group'])['0'].count().reset_index(),on=['cluster','group'])
                 .rename({'0':'weight'},axis=1)
                )
    return df, (grouped_df[(grouped_df.cluster != cl) | (grouped_df.group != group)]
            .append(new_groups)
           )

def break_group_single(df, grouped_df, cl, group):
    new_groups = df[(df.cluster == cl) & (df.group == group)].assign(group = lambda df: [str(group)+'-'+str(id) for id in range(df.shape[0])])
    
    df = (df[(df.cluster != cl) | (df.group != group)]
          .append(new_groups)
    )

    new_groups = (new_groups.groupby(['cluster','group']).max().reset_index()
                 .merge(new_groups.groupby(['cluster','group']).min().reset_index(), on=['cluster','group'],suffixes= ('_upper','_lower'))
                 .merge(new_groups.groupby(['cluster','group'])['0'].count().reset_index(),on=['cluster','group'])
                 .rename({'0':'weight'},axis=1)
                )
    return df, (grouped_df[(grouped_df.cluster != cl) | (grouped_df.group != group)]
            .append(new_groups)
           )

def break_groups_eps(df, grouped_df, errors, W, B, break_good = 'none', n_good = 100, eps=0.5):
    error_groups = grouped_df[errors.astype(np.bool)].iterrows()
    good_groups = grouped_df[~errors.astype(np.bool)].query('weight > 1')

    for id, row in error_groups:
        if row.weight < 20:
            df, grouped_df = break_group_single(df, grouped_df, row.cluster, row.group)
        else:
            df, grouped_df = break_group_eps(df, grouped_df, row.cluster, row.group, eps = eps)
    
    if break_good != 'none':
        if break_good == 'random':
            good_groups = good_groups.sample(n=min(n_good,good_groups.shape[0])).iterrows()
        elif break_good == 'largest':
            good_groups = good_groups.sort_values(by='weight',ascending=False).head(n_good).iterrows()
        elif break_good == 'closest':
            good_groups = (good_groups
                           .assign(hp_dist = computeDistance(good_groups.drop(['cluster','group','weight'],axis=1),
                                                             W, B))
                           .sort_values('hp_dist',ascending=True)
                           .head(n_good)
                           .drop('hp_dist',axis=1)
                           .iterrows()
                          )
        for id, row in good_groups:
            print('Group size %d'%row.weight)
            if row.weight < 20:
                df, grouped_df = break_group_single(df, grouped_df, row.cluster, row.group)
            elif row.weight > 200:
                df, grouped_df = break_group_eps(df, grouped_df, row.cluster, row.group, eps = eps)
            else:
                df, grouped_df = break_group_eps(df, grouped_df, row.cluster, row.group, eps = eps)


    return df, grouped_df



