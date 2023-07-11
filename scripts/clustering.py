import os
import sys
import pickle
from argparse import ArgumentParser


import numpy as np
from tqdm import tqdm

from sklearn import metrics
from sklearn.cluster import DBSCAN, KMeans
from sklearn.cluster import MiniBatchKMeans

# Setting up local details:
# This should be the location of the checkout of the FS-Mol repository:
FS_MOL_CHECKOUT_PATH = os.path.join("/data/local/apps", "Meta-Learning", "FS-Mol")
FS_MOL_DATASET_PATH = os.path.join(FS_MOL_CHECKOUT_PATH, "datasets")

os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--featurizer", type=str, default="ecfp", help="")
    args = parser.parse_args()
    return args


def kmeans_cluster(X, num_clusters):
    """
    Parameters
    ----------
    X: array-like, shape (n_samples, n_features) 
    num_clusters: int
        number of clusters
    outfile_name: str
        output file containing molecule name and cluster id
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X)
    return kmeans.cluster_centers_

def main():
    args = parse_args()
    
    output_path = os.path.join(FS_MOL_DATASET_PATH, 'embeddings', f"{args.featurizer}_test.pkl")
    with open(output_path, "rb") as f:
        data = pickle.load(f)
    
    for key, value in tqdm(data.items()):
        print(key, value.shape)
        X = value[f'{args.featurizer}']
        print(X.shape)
        kmeans_cluster(X, args.num_clusters)



if __name__ == "__main__":
    main()

