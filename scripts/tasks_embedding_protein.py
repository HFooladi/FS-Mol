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


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--esm_embeddings_path', type=str, default='data/embeddings_output', help='')
    parser.add_argument('--output_path', type=str, default='dataset/test/embedding/ecfp_pos.pkl', help='')
    parser.add_argument('--n_jobs', type=int, default=32, help='')
    parser.add_argument('--featurizer', type=str, default='', help='')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    assert args.featurizer in AVAILABLE_FEATURIZERS, f"featurizer {args.featurizer} not available. Choose from {AVAILABLE_FEATURIZERS}"





if __name__ == '__main__':
    main()