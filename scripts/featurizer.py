""" This script is used to compute the embedding for molecules in the test tasks and train tasks.
    The embedding is computed using the specified available featurizers.
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
from fs_mol.utils.distance_utils import compute_features
from tqdm import tqdm
import numpy as np
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
    "MolT5",
]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--data_fold", type=str, choices=["train", "valid", "test"], default="test", help=""
    )
    parser.add_argument("--n_jobs", type=int, default=32, help="")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    dataset = FSMolDataset.from_directory(
        FS_MOL_DATASET_PATH, task_list_file=os.path.join(FS_MOL_DATASET_PATH, "fsmol-0.1.json")
    )

    # This will create a list of all tasks in the test tasks. Each task contains `MolculeDatapoint` objects
    tasks = []
    data_fold_dict = {"train": DataFold.TRAIN, "valid": DataFold.VALIDATION, "test": DataFold.TEST}

    # next line will create iterable object that will iterate over all tasks in the test dataset
    print("reading test tasks ...")
    task_iterable = dataset.get_task_reading_iterable(data_fold_dict[args.data_fold])

    molecule_features = {}
    for task in tqdm(iter(task_iterable)):
        tasks.append(task)
        molecule_features[task.name] = {}
        for featurizer in AVAILABLE_FEATURIZERS:
            if featurizer in ["ecfp", "fcfp", "mordred"]:
                if featurizer in ["ecfp"]:
                    calc = FPCalculator(featurizer)
                    transformer = MoleculeTransformer(calc, n_jobs=args.n_jobs)
                    molecule_features[task.name][featurizer] = compute_features(task, transformer)
                else:
                    transformer = MoleculeTransformer(featurizer, n_jobs=args.n_jobs)
                    molecule_features[task.name][featurizer] = compute_features(task, transformer)

            elif featurizer in ["desc2D", "desc3D", "maccs"]:
                transformer = FPVecTransformer(kind=featurizer, dtype=float, n_jobs=args.n_jobs)
                molecule_features[task.name][featurizer] = compute_features(task, transformer)

            elif featurizer in ["pcqm4mv2_graphormer_base"]:
                transformer = GraphormerTransformer(
                    kind=featurizer, dtype=float, n_jobs=args.n_jobs
                )
                molecule_features[task.name][featurizer] = compute_features(task, transformer)

            elif featurizer in ["ChemBERTa-77M-MLM", "ChemBERTa-77M-MTR", "Roberta-Zinc480M-102M", "MolT5"]:
                transformer = PretrainedHFTransformer(
                    kind=featurizer, notation="smiles", dtype=float, n_jobs=args.n_jobs
                )
                molecule_features[task.name][featurizer] = compute_features(task, transformer)

            elif featurizer in [
                "gin_supervised_infomax",
                "gin_supervised_contextpred",
                "gin_supervised_edgepred",
                "gin_supervised_masking",
            ]:
                transformer = PretrainedDGLTransformer(
                    kind=featurizer, dtype=float, n_jobs=args.n_jobs
                )
                molecule_features[task.name][featurizer] = compute_features(task, transformer)

        molecule_features[task.name]["labels"] = torch.Tensor(
            np.array([item.bool_label for item in task.samples])
        )
        molecule_features[task.name]["smiles"] = np.array([item.smiles for item in task.samples])

    output_path = os.path.join(FS_MOL_DATASET_PATH, f"{args.data_fold}_features.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(molecule_features, f)


if __name__ == "__main__":
    main()
