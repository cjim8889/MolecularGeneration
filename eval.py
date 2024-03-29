import torch

from models.graphflowv3 import AtomGraphFlowV3
import torch.nn as nn
from rdkit import Chem
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


def get_mol(atom_map, adj_dense):

    # print(adj_t)
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
    
    return mol


if __name__ == "__main__":
    train_loader, test_loadesr = get_datasets(type="mqm9", batch_size=128)
    
    size = 128

    base = torch.distributions.Normal(loc=0., scale=1.)
    batch = next(iter(train_loader))
    print(batch.adj.shape, batch.b_adj.shape, batch.orig_adj.shape)

    # print(batch.orig_adj[0])
    network = AtomGraphFlowV3(
        mask_ratio=9,
        block_length=12,
        hidden_dim=64,
        surjection_length=4
    )

    states = torch.load("v3_12.pt", map_location=device)

    network.load_state_dict(states['model_state_dict'])


    idx = 6
    size = 16

    output_mols = []
    output_smiles = []

    for i in range(idx):
        adj = batch.adj[i: i + 1].repeat(size, 1)
        b_adj = batch.b_adj[i: i + 1].repeat(size, 1, 1)
        at_z = base.sample(sample_shape=(size, 1, 9, 7))
        orig_adj = batch.orig_adj[i: i + 1].repeat(size, 1, 1, 1)

        with torch.no_grad():
            at, _ = network.inverse(at_z, {
                "adj": adj,
                "b_adj": b_adj
            })
    
        adj_dense = orig_adj.argmax(-1)

        mols = [get_mol(at[j], adj_dense[j]) for j in range(at.shape[0])]

        print(f"processed {i}")

        smiles_list = []
        valid_mols = []
        for mol in mols:
            try:
                Chem.SanitizeMol(mol)
                smiles = Chem.MolToSmiles(mol)
                smiles_list.append(smiles)
                valid_mols.append(mol)

                print(f"Chemically Correct Molecule: {smiles}")
            except:
                continue

        output_mols += valid_mols[:8]
        output_smiles += smiles_list[:8]

    plot = Draw.MolsToGridImage(output_mols, molsPerRow=8, subImgSize=(500, 500), legends=output_smiles)
    number = np.random.randint(0, 10000)
    plot.save(f"local_interpolcation_{number}.png")

    # print(f"Validity: {t / len(mols)}")
    # print(f"U: {len(set(smiles_list)) * 1. / len(smiles_list)}")


    # print(at.shape)