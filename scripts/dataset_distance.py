## This script will return distance matrix bewteen training and test set tasks
## The distance matrix is computed based on the embedding of the training and test set tasks
## Also, test tasks hardness is computed based on the distance matrix.
import os
import sys
import pickle
from argparse import ArgumentParser

# Setting up local details:
# This should be the location of the checkout of the FS-Mol repository:
FS_MOL_CHECKOUT_PATH = os.path.join("/data/local/apps", "Meta-Learning", "FS-Mol")
FS_MOL_DATASET_PATH = os.path.join(FS_MOL_CHECKOUT_PATH, "datasets")

os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)

from fs_mol.data import FSMolDataset, DataFold
from fs_mol.utils.distance_utils import (
    compute_prototype_datamol,
    compute_task_hardness_from_distance_matrix,
)
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F

AVAILABLE_FEATURIZERS = [
    "ecfp",
    "fcfp",
    "mordred",
    "desc2D",
    "desc3D",
    "pcqm4mv2_graphormer_base",
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "Roberta-Zinc480M-102M",
    "gin_supervised_infomax",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "gin_supervised_masking",
]
AVAILABLE_METRICS = ["norm2", "cosine", "correlation", "mahalanobis", "hamming", "jaccard"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--path_train_embedding", type=str, default="train/embedding", help="")
    parser.add_argument("--path_test_embedding", type=str, default="test/embedding/", help="")
    parser.add_argument("--save_dir", type=str, default="dataset/test/hardness", help="")
    parser.add_argument("--n_jobs", type=int, default=32, help="")
    parser.add_argument("--featurizer", type=str, default="ecfp", help="")
    parser.add_argument("--metric", type=str, default="norm2", help="")
    parser.add_argument("--proportions", type=float, default=0.01, help="")
    parser.add_argument("--activity", type=str, choices=["pos", "neg"], default="pos", help="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.featurizer in AVAILABLE_FEATURIZERS, f"featurizer {args.featurizer} not available"
    assert args.metric in AVAILABLE_METRICS, f"metric {args.metric} not available"

    path_to_train_embedding = os.path.join(
        FS_MOL_DATASET_PATH, args.path_train_embedding, f"{args.featurizer}_{args.activity}.pkl"
    )
    path_to_test_embedding = os.path.join(
        FS_MOL_DATASET_PATH, args.path_test_embedding, f"{args.featurizer}_{args.activity}.pkl"
    )

    with open(path_to_train_embedding, "rb") as f:
        train_pos_prototypes = pickle.load(f)

    with open(path_to_test_embedding, "rb") as f:
        test_pos_prototypes = pickle.load(f)

    test_task_name = list(test_pos_prototypes.keys())
    train_task_name = list(train_pos_prototypes.keys())

    test_task_embedding = torch.stack(list(test_pos_prototypes.values()), dim=0)
    train_task_embedding = torch.stack(list(train_pos_prototypes.values()), dim=0)

    print("computing distance matrix")

    # compute distance matrix
    if args.metric == "norm2":
        distance_matrix = torch.cdist(train_task_embedding, test_task_embedding, p=2)
    elif args.metric == "cosine":
        similarity_matrix = F.normalize(train_task_embedding) @ F.normalize(test_task_embedding).t()
        distance_matrix = 1 - similarity_matrix

    named_distance_matrix = {
        "train_task_name": train_task_name,
        "test_task_name": test_task_name,
        "distance_matrix": distance_matrix,
    }

    # compute hardness for each test task based on distance matrix

    task_hardness = compute_task_hardness_from_distance_matrix(
        distance_matrix, proportion=args.proportions
    )

    print("saving distance matrix (task hardness))")
    # save task_hardness
    hardness_df = pd.DataFrame({"task_name": test_task_name, "hardness": task_hardness[0]})
    save_dir = os.path.join(FS_MOL_DATASET_PATH, "test", "hardness")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(
        os.path.join(save_dir, f"{args.featurizer}_{args.activity}_{args.metric}.pkl"), "wb"
    ) as f:
        pickle.dump(named_distance_matrix, f)

    hardness_df.to_csv(
        os.path.join(
            save_dir, f"{args.featurizer}_{args.activity}_{args.metric}_{args.proportions}.csv"
        ),
        index=False,
    )


if __name__ == "__main__":
    main()
