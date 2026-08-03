import pickle
import numpy as np
from sklearn import preprocessing
from functions import min_max_normalization, read_drug_list, read_drug_smiles, read_labels, read_met, read_mut, get_all_graph, get_cell_line_list, create_dataset

def process_data():
    drug_dict = read_drug_list('datasets/smile_inchi.csv')
    smile = read_drug_smiles('datasets/smile_inchi.csv', drug_dict)
    smile_graph = get_all_graph(smile)
    cell_line_dict = get_cell_line_list('datasets/cellline_listwithACH_80cellline.csv')
    cnv = pickle.load(open('datasets/512dim_copynumber.pkl', 'rb'))
    rna = pickle.load(open('datasets/512dim_RNAseq.pkl', 'rb'))
    met = read_met('datasets/METH_84cellline_378dim.csv', cell_line_dict)
    mut = read_mut('datasets/MUT_85dim_2028dim.csv', cell_line_dict)
    data = read_labels('datasets/80cell_line_ic50.csv')

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
    cnv = min_max_scaler.fit_transform(cnv)
    rna = min_max_scaler.fit_transform(rna)
    met = min_max_scaler.fit_transform(met)
    mut = min_max_scaler.fit_transform(mut)

    drug_smile = []
    cell_cnv = []
    cell_rna = []
    cell_met = []
    cell_mut = []
    targets = []

    drug_names = []
    cell_lines = []
    for item in data:
        drug, cell_line, ic50 = item
        if drug in drug_dict and cell_line in cell_line_dict:
            drug_smile.append(smile[drug_dict[drug]])
            cell_cnv.append(cnv[cell_line_dict[cell_line]])
            cell_rna.append(rna[cell_line_dict[cell_line]])
            cell_met.append(met[cell_line_dict[cell_line]])
            cell_mut.append(mut[cell_line_dict[cell_line]])
            targets.append(ic50)
            drug_names.append(drug)
            cell_lines.append(cell_line)

    targets = min_max_normalization(targets, min(targets), max(targets))

    # split data
    targets, drug_smile, cell_cnv, cell_rna, cell_mut, cell_met, drug_names, cell_lines = (
        np.asarray(targets), 
        np.asarray(drug_smile), 
        np.asarray(cell_cnv), 
        np.asarray(cell_rna), 
        np.asarray(cell_mut), 
        np.asarray(cell_met),
        np.asarray(drug_names),
        np.asarray(cell_lines)
        )

    return targets, drug_smile, cell_cnv, cell_rna, smile_graph, cell_mut, cell_met, drug_names, cell_lines

targets, drug_smile, cell_cnv, cell_rna, smile_graph, cell_mut, cell_met, drug_names, cell_lines = process_data()

for i in range(5):
    total_size = drug_smile.shape[0]
    split_0 = int(total_size * 0.2 * i)
    split_1 = split_0 + int(total_size * 0.1)
    split_2 = int(total_size * 0.2 * (i+1))

    ds_test = drug_smile[split_0:split_1]
    ds_val = drug_smile[split_1:split_2]
    ds_train = np.concatenate((drug_smile[:split_0],drug_smile[split_2:]), axis=0)

    cnv_test = cell_cnv[split_0:split_1]
    cnv_val = cell_cnv[split_1:split_2]
    cnv_train = np.concatenate((cell_cnv[:split_0],cell_cnv[split_2:]), axis=0)

    rna_test = cell_rna[split_0:split_1]
    rna_val = cell_rna[split_1:split_2]
    rna_train = np.concatenate((cell_rna[:split_0],cell_rna[split_2:]), axis=0)

    met_test = cell_met[split_0:split_1]
    met_val = cell_met[split_1:split_2]
    met_train = np.concatenate((cell_met[:split_0],cell_met[split_2:]), axis=0)

    mut_test = cell_mut[split_0:split_1]
    mut_val = cell_mut[split_1:split_2]
    mut_train = np.concatenate((cell_mut[:split_0],cell_mut[split_2:]), axis=0)

    targ_test = targets[split_0:split_1]
    targ_val = targets[split_1:split_2]
    targ_train = np.concatenate((targets[:split_0],targets[split_2:]), axis=0)

    drug_names_test = drug_names[split_0:split_1]
    drug_names_val = drug_names[split_1:split_2]
    drug_names_train = np.concatenate((drug_names[:split_0],drug_names[split_2:]), axis=0)

    cell_lines_test = cell_lines[split_0:split_1]
    cell_lines_val = cell_lines[split_1:split_2]
    cell_lines_train = np.concatenate((cell_lines[:split_0],cell_lines[split_2:]), axis=0)

    train_dataset = create_dataset(
        ds_train,
        cnv_train,
        rna_train,
        targ_train,
        met_train,
        mut_train,
        smile_graph,
        drug_names_train,
        cell_lines_train
    )

    val_dataset = create_dataset(
        ds_val,
        cnv_val,
        rna_val,
        targ_val,
        met_val,
        mut_val, 
        smile_graph,
        drug_names_val,
        cell_lines_val
    )

    test_dataset = create_dataset(
        ds_test,
        cnv_test,
        rna_test,
        targ_test,
        met_test,
        mut_test,
        smile_graph,
        drug_names_test,
        cell_lines_test
    )

    # Save as pickles
    with open(f'cross-val/train_fold_{i}.pkl', 'wb') as f:
        pickle.dump(train_dataset, f)

    with open(f'cross-val/validation_fold_{i}.pkl', 'wb') as f:
        pickle.dump(val_dataset, f)

    with open(f'cross-val/test_fold_{i}.pkl', 'wb') as f:
        pickle.dump(test_dataset, f)
