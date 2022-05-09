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
bond_dict = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE, 4: Chem.rdchem.BondType.AROMATIC}
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
            if adj_dense[i, j] == 0:
                continue

            
            bond_rdkit = bond_dict[int(adj_dense[i, j])]
            mol.AddBond(i, j, bond_rdkit)
    
    return mol

if __name__ == '__main__':
    
    # size = 256

    # atomflow_states = torch.load("atom.pt", map_location=device)
    # atomflow = AtomGraphFlow(block_length=6, mask_ratio=9.)
    # atomflow.load_state_dict(atomflow_states['model_state_dict'])

    # adjflow_states = torch.load("adj.pt", map_location=device)
    # adjflow = ArgmaxFlow(t=6, mask_ratio=9., max_nodes=9, hidden_dim=64)
    # adjflow.load_state_dict(adjflow_states['model_state_dict'])

    # base = torch.distributions.Normal(loc=0., scale=1.)

    # z = base.sample(sample_shape=(size, 1, 45, 5))
    # adj, _ = adjflow.inverse(z)
    # adj = adj.squeeze(1)

    # print(adj.shape, "\n\n")

    # indices = torch.triu_indices(9, 9)
    # adj_dense = torch.zeros(size, 9, 9, dtype=torch.int64, device=device)

    # adj_dense[..., indices[0], indices[1]] = adj

    # adj_dense += adj_dense.clone().transpose(1, 2)
    # adj_dense[..., torch.arange(adj_dense.shape[1]), torch.arange(adj_dense.shape[2])] = 0
    # print(adj_dense.shape)

    # b_adj = adj_dense.clone()
    # b_adj[b_adj > 0.] = 1.



    # at_z = base.sample(sample_shape=(size, 1, 9, 7))
    # at, _ = atomflow.inverse(at_z, {
    #     "adj": adj,
    #     "b_adj": b_adj
    # })

    # print(at.shape, "\n\n")

    # # # print(adj[0], at[0])

    # print("processing...")
    # mols = [get_mol(at[i], adj_dense[i]) for i in range(at.shape[0])]
    
    # print("processed")
    # t = 0

    # smiles_list = []
    # for mol in mols:
    #     try:
    #         Chem.SanitizeMol(mol)
    #         smiles = Chem.MolToSmiles(mol)
    #         smiles_list.append(smiles)
    #         print(f"Correct Molecule: {smiles}")
    #         t += 1
    #     except:
    #         continue


    transform = T.Compose([ToDenseAdjV2(num_nodes=9)])
    dataset = ModifiedQM9(root="./mqm9-datasets", pre_transform=transform)

    # for data in dataset:
    #     print(f"""{data.smiles} \n\n 
    #     {data.orig_adj.argmax(-1)} \n\n
    #     {data.adj} \n\n
    #     {data.b_adj} \n\n
    #     {data.x} \n\n
    #     {data.bond_num}""")
    


    mols_qm9 = []
    smiles_qm9 = set()
    for data in dataset:
        at = data.x.long()
        adj = torch.cat((torch.zeros(9, 9, 1, dtype=torch.int64, device=device), data.orig_adj), dim=-1).argmax(-1)

        mols_qm9.append(get_mol(at, adj))

    for mol in mols_qm9:
        try:
            Chem.SanitizeMol(mol)
            smiles = Chem.MolToSmiles(mol)
            smiles_qm9.add(smiles)
            print(f"Correct Molecule: {smiles}")
        except:
            continue

    
    # print(len(smiles_qsm9))
    # existed = 0

    # for s in smiles_list:
        # if s in smiles_qm9:
            # existed += 1

    # print(f"N: {existed / len(smiles_list)}")
    # print(f"V: {t * 1. / len(mols)}")
    # print(f"U: {len(set(smiles_list)) * 1. / len(smiles_list)}")