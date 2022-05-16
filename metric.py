import torch

from models.graphflowv3 import AtomGraphFlowV3
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import QED

from rdkit.Chem import Draw


import numpy as np

from utils import get_datasets

device = torch.device("cpu")

context_size=16
num_classes=5
embedding_dim=7
hidden_dim=64

bond_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}

atom_decoder = ['B', 'H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}

'''
QED of the QM9: 0.4540475089608309
Validity: 0.765625
Uniqueness: 0.9948979591836735
Novelty: 0.7959183673469388
QED of the Generated: 0.5070806336697697

Validity: 0.8385817307692307
Uniqueness: 0.9829129373474369
Novelty: 0.838893409275834
QED of the Generated: 0.5012164550759273

Validity: 0.8421875
Uniqueness: 0.9964912280701754
Novelty: 0.8263157894736842
QED of the Generated: 0.48588555810765555
Max QED of the Generated: 0.6537246457346916
90 percentile QED of the Generated: 0.5702465578000752
'''

def get_mol(atom_map, adj_dense, verbose=True):
    mol = Chem.RWMol()

    end_index = -1
    for atom_index in range(atom_map.shape[0]):
        if atom_map[atom_index, 0] == 0:
            end_index = atom_index
            break

        mol.AddAtom(Chem.Atom(atom_decoder[atom_map[atom_index, 0]]))
    
    if end_index == -1:
        end_index = 9


    for i in range(end_index):
        for j in range(i + 1, end_index):
            if adj_dense[i, j] == 4:
                continue

            
            bond_rdkit = bond_dict[int(adj_dense[i, j])]
            mol.AddBond(i, j, bond_rdkit)
    
    try:
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol)

        if verbose:
            print(f"Chemically Correct Molecule: {smiles}")

        return mol, smiles
    except:
        return mol, None
    
def get_smiles_from_loader(loader):
    smiles_list = []
    mols_list = []
    for batch in loader:
        adj_dense = batch.orig_adj.argmax(dim=-1)
        x = batch.x.long()
        for idx in range(batch.x.shape[0]):
            mol, smile = get_mol(x[idx], adj_dense[idx], verbose=False)
            if smile is not None:
                smiles_list.append(smile)
                mols_list.append(mol)

    return mols_list, smiles_list

if __name__ == "__main__":
    batch_size = 128
    batch_number = 6
    calculate_qm9 = False

    train_loader, test_loadesr = get_datasets(type="mqm9", batch_size=batch_size)

    network = AtomGraphFlowV3(
        mask_ratio=9,
        block_length=12,
        hidden_dim=64,
        surjection_length=4
    )

    states = torch.load("v3_12.pt", map_location=device)
    network.load_state_dict(states['model_state_dict'])
    
    base = torch.distributions.Normal(loc=0., scale=1.)

    output_mols = []

    idx = 0
    for batch in test_loadesr:
        if idx >= batch_number:
            break
        
        at_z = base.sample(sample_shape=(batch_size, 1, 9, 7))

        with torch.no_grad():
            at, _ = network.inverse(at_z, {
                "adj": batch.adj,
                "b_adj": batch.b_adj
            })
        
        adj_dense = batch.orig_adj.argmax(-1)

        mols = [get_mol(at[j], adj_dense[j]) for j in range(at.shape[0])]
        output_mols += mols

        print(f"processed batch {idx}")
        idx += 1


    valid_smiles = []
    valid_mols = []

    for mol in output_mols:
        if mol[1] is None:
            continue

        valid_mols.append(mol[0])
        valid_smiles.append(mol[1])

    validity = len(valid_smiles) / len(output_mols)
    uniqueness = len(set(valid_smiles)) / len(valid_smiles)

    print("Loading from QM9...")
    mols, smiles = get_smiles_from_loader(train_loader)
    print("QM9 loaded")

    if calculate_qm9:
        qed_qm9 = [QED.qed(mol) for mol in mols]
        print(f"QED of the QM9: {np.mean(qed_qm9)}")


    unique_smiles = set(smiles)

    novel_smiles = [s for s in valid_smiles if s not in unique_smiles]

    qed = [QED.qed(mol) for mol in valid_mols]

    novelty = len(novel_smiles) / len(valid_smiles)

    print(f"Validity: {validity}")
    print(f"Uniqueness: {uniqueness}")
    print(f"Novelty: {novelty}")
    print(f"QED of the Generated: {np.mean(qed)}")
    print(f"Max QED of the Generated: {np.max(qed)}")
    print(f"90 percentile QED of the Generated: {np.percentile(qed, 90)}")

