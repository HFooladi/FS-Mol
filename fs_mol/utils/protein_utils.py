import os
from chembl_webresource_client.new_client import new_client
import requests as r
from Bio import SeqIO
from io import StringIO
from pathlib import Path
import torch
from tqdm import tqdm
from typing import List, Tuple, Union


def get_protein_accession(target_chembl_id: str) -> Union[str, None]:
    """Returns the target protein accesion id for a given target chembl id.
    This id can be used to retrieve the protein sequence from the UniProt.

    Args:
        target_chembl_id: Chembl id of the target.
    """
    target = new_client.target
    target_result = target.get(target_chembl_id)
    if "target_components" in target_result:
        return target_result["target_components"][0]["accession"]
    else:
        return None


def get_target_chembl_id(assay_chembl_id: str) -> Union[str, None]:
    """Returns the target chembl id for a given assay chembl id.

    Args:
        assay_chembl_id: Chembl id of the assay.
    """
    assay = new_client.assay
    assay_result = assay.get(assay_chembl_id)
    if "target_chembl_id" in assay_result:
        target_chembl_id = assay_result["target_chembl_id"]
        return target_chembl_id
    return None


def get_protein_sequence(protein_accession: str) -> List[SeqIO.SeqRecord]:
    """Returns the protein sequence for a given protein accession id.

    Args:
        protein_accession: Accession id of the protein.
    """
    cID = protein_accession
    baseUrl = "http://www.uniprot.org/uniprot/"
    currentUrl = baseUrl + cID + ".fasta"
    response = r.post(currentUrl)
    cData = "".join(response.text)

    Seq = StringIO(cData)
    pSeq = list(SeqIO.parse(Seq, "fasta"))
    return pSeq


def read_esm_embedding(fs_mol_dataset_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reads the ESM embedding from a given path.

    Args:
        fs_mol_dataset_path: Path to the FS_MOL Dataset.
    """
    ESM_EMBEDDING_PATH = os.path.join(fs_mol_dataset_path, "targets", "esm2_output")

    train_esm = os.path.join(ESM_EMBEDDING_PATH, "train_proteins")
    valid_esm = os.path.join(ESM_EMBEDDING_PATH, "valid_proteins")
    test_esm = os.path.join(ESM_EMBEDDING_PATH, "test_proteins")

    train_files = Path(train_esm).glob("*")
    valid_files = Path(valid_esm).glob("*")
    test_files = Path(test_esm).glob("*")

    train_emb: List = []
    valid_emb: List = []
    test_emb: List = []

    train_emb_tensor = torch.empty(0)
    valid_emb_tensor = torch.empty(0)
    test_emb_tensor = torch.empty(0)

    for file in tqdm(train_files):
        train_emb.append(torch.load(file))
        train_emb_tensor = torch.cat(
            (train_emb_tensor, train_emb[-1]["mean_representations"][33][None, :]), 0
        )

    for file in tqdm(valid_files):
        valid_emb.append(torch.load(file))
        valid_emb_tensor = torch.cat(
            (valid_emb_tensor, valid_emb[-1]["mean_representations"][33][None, :]), 0
        )

    for file in tqdm(test_files):
        test_emb.append(torch.load(file))
        test_emb_tensor = torch.cat(
            (test_emb_tensor, test_emb[-1]["mean_representations"][33][None, :]), 0
        )

    assert train_emb_tensor.shape[0] == len(train_emb)
    assert valid_emb_tensor.shape[0] == len(valid_emb)
    assert test_emb_tensor.shape[0] == len(test_emb)

    return train_emb_tensor, valid_emb_tensor, test_emb_tensor
