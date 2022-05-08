import torch
from models import atomflow


from models.atomflow import AtomFlow
from models.argmaxflowv2 import ArgmaxFlow

from rdkit import Chem
from torch_geometric.nn import DenseGraphConv

device = torch.device("cpu")
bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
atom_decoder = ['B', 'H', 'C', 'N', 'O', 'F']
atom_encoder = {'H': 1, 'C': 2, 'N': 3, 'O': 4, 'F': 5}


def get_mol(atom_map, adj):

    adj_t = torch.zeros(9, 9, dtype=torch.int64, device=device)
    indices = torch.tril_indices(row=9, col=9, offset=0)

    adj_t[indices[0], indices[1]] = adj
    adj_t[torch.arange(adj_t.shape[0]), torch.arange(adj_t.shape[0])] = 0

    # print(adj_t)
    mol = Chem.RWMol()

    ignore_idx = []
    for atom_index in range(atom_map.shape[0]):
        if atom_map[atom_index, 0] == 0:
            ignore_idx.append(atom_index)
            continue

        if adj_t[:, atom_index].sum() == 0:
            ignore_idx.append(atom_index)
            continue
            

        mol.AddAtom(Chem.Atom(atom_decoder[atom_map[atom_index, 0]]))

    print(ignore_idx)
    for i in range(atom_map.shape[0]):
        for j in range(i + 1, atom_map.shape[0]):
            if adj_t[j, i] == 0:
                continue

            if i in ignore_idx or j in ignore_idx:
                continue

            bond_rdkit = bond_dict[int(adj_t[j, i])]
            mol.AddBond(j, i, bond_rdkit)
    
    return mol

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

    print(adj.shape, "\n\n")

    at_z = base.sample(sample_shape=(1, 1, 9, 7))
    at, _ = atomflow.inverse(at_z, adj)

    print(at.shape, "\n\n")

    # print(adj[0], at[0])

    graphconv = DenseGraphConv(7, 14)
    out = graphconv(torch.randn(1, 9, 7), torch.zeros(1, 9, 9))
    print(out.shape)
    # print("processing...")
    # mols = [get_mol(at[i], adj[i]) for i in range(at.shape[0])]

    # print("processed")
    # t = 0

    # for mol in mols:
    #     try:
    #         Chem.SanitizeMol(mol)
    #         print("Correct Molecule")
    #         t += 1
    #     except:
    #         continue

    # print(t * 1. / len(mols))