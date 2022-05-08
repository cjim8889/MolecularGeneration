import torch
from models import atomflow


from models.atomflow import AtomFlow
from models.argmaxflowv2 import ArgmaxFlow

from rdkit import Chem


device = torch.device("cpu")
bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['NO','H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}

if __name__ == '__main__':
    
    # transform = T.Compose([ToDenseAdjV2(num_nodes=9)])
    # dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)

    atomflow_states = torch.load("atom.pt", map_location=device)
    atomflow = AtomFlow(block_length=6, mask_ratio=9.)
    atomflow.load_state_dict(atomflow_states['model_state_dict'])

    adjflow_states = torch.load("adj.pt", map_location=device)
    adjflow = ArgmaxFlow(t=6, mask_ratio=9., max_nodes=9, hidden_dim=64)
    adjflow.load_state_dict(adjflow_states['model_state_dict'])

    base = torch.distributions.Normal(loc=0., scale=1.)

    z = base.sample(sample_shape=(1, 1, 45, 5))
    adj, _ = adjflow.inverse(z)
    adj = adj.squeeze(1)

    print(adj.shape, "\n\n", adj[0])

    at_z = base.sample(sample_shape=(1, 9, 7))
    at, _ = atomflow.inverse(at_z, adj)

    print(at.shape, "\n\n", at[0])

    atom_map = at[0]


    adj_t = torch.zeros(9, 9, dtype=torch.int64, device=device)
    indices = torch.triu_indices(row=9, col=9, offset=0)
    adj_t[indices[0], indices[1]] = adj[0]

    mol = Chem.RWMol()

    for atom_index in range(atom_map.shape[0]):
        if atom_map[atom_index, 0] == 0:
            continue

        mol.AddAtom(Chem.Atom(atom_decoder[atom_map[atom_index, 0]]))

    for i in range(atom_map.shape[0]):
        for j in range(i + 1, atom_map.shape[0]):
            if adj_t[i, j] == 0:
                continue

            bond_rdkit = bond_dict[int(adj_t[i, j])]
            mol.AddBond(i, j, bond_rdkit)
            print(i, j, bond_rdkit)

    mol = Chem.SanitizeMol(mol)
    smiles = Chem.MolToSmiles(mol)
    print(smiles)