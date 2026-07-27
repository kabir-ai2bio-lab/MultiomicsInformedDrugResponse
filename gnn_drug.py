import torch
import csv
import numpy as np
from rdkit import Chem
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GCNConv, JumpingKnowledge, global_max_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# load datasets
def read_drug_list(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_dict = {}
    index = 0
    for line in reader:
        drug_dict[line[3]] = index
        index += 1
    return drug_dict

def read_drug_smiles(filename, drug_dict):  # load drugs' SMILES
    f = open(filename, 'r')
    reader = csv.reader(f)
    reader.__next__()
    drug_smiles = [list() for i in range(len(drug_dict))]
    for line in reader:
        drug_smiles[drug_dict[line[3]]] = line[9]  # use the index in dictionary
    return drug_smiles

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def get_all_graph(drug_smiles):
    smile_graph = {}
    for smile in drug_smiles:
        if len(smile) > 0:
            smile_graph[smile] = smile_to_graph(smile)
    return smile_graph

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetValence(Chem.ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])

class GNN_drug(torch.nn.Module):
    def __init__(self, layer_drug, dim_drug):
        super().__init__()
        self.layer_drug = layer_drug
        self.dim_drug = dim_drug
        self.JK = JumpingKnowledge('cat')
        self.convs_drug = torch.nn.ModuleList()
        self.bns_drug = torch.nn.ModuleList()
        self.gcn1 = GCNConv(78,self.dim_drug)
        self.gcn2 = GCNConv(self.dim_drug,self.dim_drug)
        self.relu = nn.ReLU()
        for i in range(self.layer_drug):
            if i:
               block = nn.Sequential(nn.Linear(self.dim_drug, self.dim_drug), nn.ReLU(),
                                      nn.Linear(self.dim_drug, self.dim_drug))
            else:
               block = nn.Sequential(nn.Linear(78, self.dim_drug), nn.ReLU(), nn.Linear(self.dim_drug, self.dim_drug))
            conv = GINConv(block)
            bn = torch.nn.BatchNorm1d(self.dim_drug)

            self.convs_drug.append(conv)
            self.bns_drug.append(bn)

    def forward(self, drug):
        x, edge_index, batch = drug.x, drug.edge_index, drug.batch
        x_drug_list = []
        x_drug_m = 1
        x_drug_e = 0
        x_gcn  = self.gcn1(x,edge_index)
        x_gcn1 = self.relu(x_gcn)
        x_gcn2 = self.gcn2(x_gcn1,edge_index)
        x_gcn2_ = self.relu(x_gcn2)
        x_gcn2_a = x_gcn2_+x_gcn1
        x_gcn2_m = x_gcn2_*x_gcn1
        
        for i in range(self.layer_drug):
            x = F.relu(self.convs_drug[i](x, edge_index))
            x = self.bns_drug[i](x)
            x_drug_m*=x
            x_drug_e+=x
            x_drug_list.append(x)

        node_representation = self.JK(x_drug_list)
        node_representation=torch.cat([node_representation,x_drug_m,x_drug_e,x_gcn1,x_gcn2_,x_gcn2_a,x_gcn2_m],dim=-1)
        x_drug = global_max_pool(node_representation, batch)
        return x_drug

def read_process_data(filename):
    drug_dict = read_drug_list(filename)
    smiles = read_drug_smiles(filename, drug_dict)
    smile_graph = get_all_graph(smiles)
    
    drug_graphs = []

    for smile in smiles:
        if smile in smile_graph:

            c_size, features, edge_index = smile_graph[smile]

            graph = Data(
                x = torch.tensor(np.asarray(features), dtype=torch.float),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            )

            graph.smile = smile

            drug_graphs.append(graph)

    loader = DataLoader(
        drug_graphs,
        batch_size=32,
        shuffle=False
    )

    model = GNN_drug(3, 128)
    model.eval()
    drug_features = []

    with torch.no_grad():
        for drug in loader:
            feature = model(drug)
            drug_features.append(feature)

    drug_features = torch.cat(drug_features, dim=0)

    return drug_features

drug_features = read_process_data('datasets/smile_inchi.csv')
torch.save(drug_features, "datasets/drug_features.pt")


