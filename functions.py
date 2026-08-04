import numpy as np
import csv
import random
import torch
import random
import numpy as np
from torch_geometric.data import Data
from model.gnn_drug import smile_to_graph



def safe_mape(actuals, predictions, min_value=0.01):
    actuals = np.maximum(actuals, min_value)
    return np.mean(np.abs((actuals - predictions) / actuals)) * 100

def min_max_normalization(list, min, max):
    res = []
    for item in list:
        temp = (item - min) / (max - min)
        res.append(temp)
    return res

def read_drug_list(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        drug_dict = {}
        index = 0
        for line in reader:
            drug_dict[line[3]] = index
            index += 1
    return drug_dict

def read_drug_smiles(filename, drug_dict):  # load drugs' SMILES
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        drug_smiles = [list() for i in range(len(drug_dict))]
        for line in reader:
            drug_smiles[drug_dict[line[3]]] = line[9]  # use the index in dictionary

    return drug_smiles

def get_cell_line_list(filename):  # load cell lines and build a dictionary
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        cell_line_dict = {}
        index = 0
        for line in reader:
            cell_line_dict[line[0]] = index
            index += 1

    return cell_line_dict

def read_labels(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = []
        for line in reader:
            drug = line[0]
            cell_line = line[2]
            ic50 = float(line[7])
            data.append((drug, cell_line, ic50))
        random.shuffle(data)

    return data

def read_mut(filename, cell_line_dict):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        mut = [list() for _ in range(len(cell_line_dict))]
        for line in reader:
            if line[0] in cell_line_dict:
                mut[cell_line_dict[line[0]]] = line[1:]

    return mut

def read_met(filename, cell_line_dict):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        met = [list() for _ in range(len(cell_line_dict))]
        for line in reader:
            if line[0] in cell_line_dict:
                met[cell_line_dict[line[0]]] = line[1:]

    return met

def get_all_graph(drug_smiles):
    smile_graph = {}
    for smile in drug_smiles:
        if len(smile) > 0:
            graph = smile_to_graph(smile)
            smile_graph[smile] = graph

    return smile_graph

def create_dataset(drugs, cnvs, rnas, targets, mets, muts, smile_graph, drug_name, cell_line):
    dataset = []

    for drug, cnv, rna, ic50_targ, met, mut, drug_id, cell_id in zip(
        drugs,
        cnvs, 
        rnas, 
        targets, 
        mets, 
        muts,
        drug_name,
        cell_line):
        c_size, features, edge_index = smile_graph[drug]

        data = Data(
            x=torch.tensor(np.asarray(features), dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        )

        data.cnv = torch.tensor(cnv, dtype=torch.float).reshape(1,512)
        data.rna = torch.tensor(rna, dtype=torch.float).reshape(1,512)
        data.met = torch.tensor(met, dtype=torch.float).reshape(1,378)
        data.mut = torch.tensor(mut, dtype=torch.float).reshape(1,2028)
        data.ic50_targ = torch.tensor([[ic50_targ]], dtype=torch.float)

        data.drug_name = drug_id
        data.cell_line = cell_id

        dataset.append(data)

    return dataset