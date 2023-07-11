""" This script is used to compute the embedding for molecules in the test tasks and train tasks.
    The embedding is computed using the specified available featurizers. Then for each featurizer, a dictionary
    is created that maps the task name to the embedding of the molecules in that task. The dictionary is like the following:
    {'CheMBL1234:{'ecfp': torch.Tensor, 'labels': torch.Tensor, 'smiles': np.array}, 'CheMBL5678': {'ecfp': torch.Tensor, 'labels': torch.Tensor, 'smiles': np.array}, ...}
    The dictionary is then saved to a pickle file (for train and test tasks separately).
"""

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
from fs_mol.utils.distance_utils import compute_features_smiles_labels
from tqdm import tqdm
import pandas as pd
import torch
import datamol as dm
from molfeat.calc import FPCalculator
from molfeat.trans.fp import FPVecTransformer
from molfeat.trans import MoleculeTransformer
from molfeat.trans.pretrained import GraphormerTransformer, PretrainedDGLTransformer
from molfeat.trans.pretrained.hf_transformers import PretrainedHFTransformer


AVAILABLE_FEATURIZERS = [
    "ecfp",
    "mordred",
    "desc2D",
    "maccs",
    "ChemBERTa-77M-MLM",
    "ChemBERTa-77M-MTR",
    "Roberta-Zinc480M-102M",
    "gin_supervised_infomax",
    "gin_supervised_contextpred",
    "gin_supervised_edgepred",
    "gin_supervised_masking",
    ] 



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--esm_embeddings_path", type=str, default="data/embeddings_output", help=""
    )
    parser.add_argument(
        "--output_path", type=str, default="dataset/test/embedding/ecfp_pos.pkl", help=""
    )
    parser.add_argument("--n_jobs", type=int, default=32, help="")
    parser.add_argument("--featurizer", type=str, default="", help="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = FSMolDataset.from_directory(
        FS_MOL_DATASET_PATH, task_list_file=os.path.join(FS_MOL_DATASET_PATH, "fsmol-0.1.json")
    )

    # This will create a list of all tasks in the test tasks. Each task contains `MolculeDatapoint` objects
    test_tasks = []
    # next line will create iterable object that will iterate over all tasks in the test dataset
    print("reading test tasks ...")
    test_task_iterable = dataset.get_task_reading_iterable(DataFold.TEST)
    for task in tqdm(iter(test_task_iterable)):
        test_tasks.append(task)

    print("reading train tasks ...")
    train_tasks = []
    train_task_iterable = dataset.get_task_reading_iterable(DataFold.TRAIN)
    for task in tqdm(iter(train_task_iterable)):
        train_tasks.append(task)

    for featurizer in tqdm(AVAILABLE_FEATURIZERS):
        if featurizer in ["ecfp", "fcfp", "mordred"]:
            if featurizer in ["ecfp"]:
                calc = FPCalculator(featurizer)
                transformer = MoleculeTransformer(calc, n_jobs=args.n_jobs)
            else:
                transformer = MoleculeTransformer(featurizer, n_jobs=args.n_jobs)

        elif featurizer in ["desc2D", "desc3D", "maccs"]:
            transformer = FPVecTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)

        elif featurizer in ["pcqm4mv2_graphormer_base"]:
            transformer = GraphormerTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)

        elif featurizer in ["ChemBERTa-77M-MLM", "ChemBERTa-77M-MTR", "Roberta-Zinc480M-102M"]:
            transformer = PretrainedHFTransformer(
                kind=featurizer, notation="smiles", dtype=float, n_jobs=args.n_jobs
            )

        elif featurizer in [
            "gin_supervised_infomax",
            "gin_supervised_contextpred",
            "gin_supervised_edgepred",
            "gin_supervised_masking",
        ]:
            transformer = PretrainedDGLTransformer(
                kind=featurizer, dtype=float, n_jobs=args.n_jobs
            )

        print("computing prototypes for test tasks ...")
        test_features = [compute_features_smiles_labels(task, transformer) for task in tqdm(test_tasks)]

        print("computing prototypes for train tasks ...")
        train_features = [compute_features_smiles_labels(task, transformer) for task in tqdm(train_tasks)]

        features_test = {
            test_tasks[i].name: {featurizer: prototypes[0], 'labels': prototypes[1], 'smiles': prototypes[2]} for i, prototypes in enumerate(test_features)
        }

        features_train = {
            train_tasks[i].name: {featurizer: prototypes[0], 'labels': prototypes[1], 'smiles': prototypes[2]} for i, prototypes in enumerate(train_features)
        }

        path_to_save_embedding_test = os.path.join(
            FS_MOL_DATASET_PATH, 'embeddings', f"{featurizer}_test.pkl"
        )

        path_to_save_embedding_train = os.path.join(
            FS_MOL_DATASET_PATH, 'embeddings', f"{featurizer}_train.pkl"
        )     

        with open(path_to_save_embedding_test, "wb") as f:
            pickle.dump(features_test, f)

        with open(path_to_save_embedding_train, "wb") as f:
            pickle.dump(features_train, f)


if __name__ == "__main__":
    main()
