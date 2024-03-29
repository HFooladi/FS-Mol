import os
import sys
import pickle
from argparse import ArgumentParser

# Setting up local details:
# This should be the location of the checkout of the FS-Mol repository:
FS_MOL_CHECKOUT_PATH = os.path.join("/home/vscholz/Documents/hfooladi", "Meta-Learning", "FS-MOL")
FS_MOL_DATASET_PATH = os.path.join(FS_MOL_CHECKOUT_PATH, "datasets")

os.chdir(FS_MOL_CHECKOUT_PATH)
sys.path.insert(0, FS_MOL_CHECKOUT_PATH)


from fs_mol.data import FSMolDataset, DataFold
from fs_mol.utils.distance_utils import compute_prototype_datamol
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
    assert (
        args.featurizer in AVAILABLE_FEATURIZERS
    ), f"featurizer {args.featurizer} not available. Choose from {AVAILABLE_FEATURIZERS}"
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

    if args.featurizer in ["ecfp", "fcfp", "mordred"]:
        if args.featurizer in ["ecfp"]:
            calc = FPCalculator(args.featurizer)
            transformer = MoleculeTransformer(calc, n_jobs=args.n_jobs)
        else:
            transformer = MoleculeTransformer(args.featurizer, n_jobs=args.n_jobs)

    elif args.featurizer in ["desc2D", "desc3D"]:
        transformer = FPVecTransformer(kind=args.featurizer, dtype=float, n_jobs=args.n_jobs)

    elif args.featurizer in ["pcqm4mv2_graphormer_base"]:
        transformer = GraphormerTransformer(kind=args.featurizer, dtype=float, n_jobs=args.n_jobs)

    elif args.featurizer in ["ChemBERTa-77M-MLM", "ChemBERTa-77M-MTR", "Roberta-Zinc480M-102M"]:
        transformer = PretrainedHFTransformer(
            kind=args.featurizer, notation="smiles", dtype=float, n_jobs=args.n_jobs
        )

    elif args.featurizer in [
        "gin_supervised_infomax",
        "gin_supervised_contextpred",
        "gin_supervised_edgepred",
        "gin_supervised_masking",
    ]:
        transformer = PretrainedDGLTransformer(
            kind=args.featurizer, dtype=float, n_jobs=args.n_jobs
        )

    print("computing prototypes for test tasks ...")
    test_prototypes = [compute_prototype_datamol(task, transformer) for task in tqdm(test_tasks)]

    print("computing prototypes for train tasks ...")
    train_prototypes = [compute_prototype_datamol(task, transformer) for task in tqdm(train_tasks)]

    neg_test_prototypes = {
        test_tasks[i].name: prototypes[0] for i, prototypes in enumerate(test_prototypes)
    }
    pos_test_prototypes = {
        test_tasks[i].name: prototypes[1] for i, prototypes in enumerate(test_prototypes)
    }

    neg_train_prototypes = {
        train_tasks[i].name: prototypes[0] for i, prototypes in enumerate(train_prototypes)
    }
    pos_train_prototypes = {
        train_tasks[i].name: prototypes[1] for i, prototypes in enumerate(train_prototypes)
    }

    path_to_save_embedding_test_pos = os.path.join(
        FS_MOL_DATASET_PATH, "test", "embedding", f"{args.featurizer}_pos.pkl"
    )
    path_to_save_embedding_test_neg = os.path.join(
        FS_MOL_DATASET_PATH, "test", "embedding", f"{args.featurizer}_neg.pkl"
    )
    path_to_save_embedding_train_pos = os.path.join(
        FS_MOL_DATASET_PATH, "train", "embedding", f"{args.featurizer}_pos.pkl"
    )
    path_to_save_embedding_train_neg = os.path.join(
        FS_MOL_DATASET_PATH, "train", "embedding", f"{args.featurizer}_neg.pkl"
    )

    with open(path_to_save_embedding_test_pos, "wb") as f:
        pickle.dump(pos_test_prototypes, f)

    with open(path_to_save_embedding_test_neg, "wb") as f:
        pickle.dump(neg_test_prototypes, f)

    with open(path_to_save_embedding_train_pos, "wb") as f:
        pickle.dump(pos_train_prototypes, f)

    with open(path_to_save_embedding_train_neg, "wb") as f:
        pickle.dump(neg_train_prototypes, f)


if __name__ == "__main__":
    main()
