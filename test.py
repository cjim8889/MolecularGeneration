import torch
from models import atomflow


from models.atomflow.graphflow import AtomGraphFlow
from models.argmaxflowv2 import ArgmaxFlow

from rdkit import Chem
from torch_geometric.nn import DenseGraphConv

from utils import ToDenseAdjV2
from utils import ModifiedQM9

from torch_geometric import transforms as T

device = torch.device("cpu")
bond_dict = {0: Chem.rdchem.BondType.SINGLE, 1: Chem.rdchem.BondType.DOUBLE, 2: Chem.rdchem.BondType.TRIPLE, 3: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['B', 'H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}


# AtomgGraph V: 17.9 U: 73.1 N: 50.1
# N: 0.3076923076923077
# V: 0.25390625
# U: 0.7384615384615385
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

if __name__ == '__main__':
    
    pass