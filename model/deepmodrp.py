import torch
import torch.nn as nn
from gnn_drug import GNN_drug
from torch_geometric.loader import DataLoader
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from statistics import mean
from functions import safe_mape
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

class DeepMoDRP(torch.nn.Module):
    def __init__(self, n_filters=4, output_dim=256, dropout=0.2):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()

        self.layer_drug = 3
        self.dim_drug = 128

        # GNN Drug
        self.GNN_drug = GNN_drug(self.layer_drug, self.dim_drug)
        self.drug_emb = nn.Sequential(
            nn.Linear(self.dim_drug * (self.layer_drug+6), 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # cnv
        self.cnv_ln = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # rna
        self.rna_ln = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),

            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # met
        self.met_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(in_channels=n_filters, out_channels=n_filters* 2, kernel_size=8),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            nn.MaxPool1d(3),
        )
        self.met_ln = nn.Sequential(
            nn.Linear(160, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # mut
        self.mut_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8),
            nn.BatchNorm1d(n_filters),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(in_channels=n_filters, out_channels=n_filters* 2, kernel_size=8),
            nn.BatchNorm1d(n_filters * 2),
            nn.ReLU(),
            nn.MaxPool1d(3),

            nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8),
            nn.BatchNorm1d(n_filters*4),
            nn.ReLU(),
            nn.MaxPool1d(3)
        )
        self.mut_ln = nn.Sequential(
            nn.Linear(1136, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, output_dim),
            nn.BatchNorm1d(output_dim)
        )

        # fusion/prediction
        self.comb = nn.Sequential(
            nn.Linear(5 * output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(128, 1)
        )

    def forward(self, data):
        cnv = data.cnv
        rna = data.rna
        met = data.met.unsqueeze(1)
        mut = data.mut.unsqueeze(1)
        x_drug = self.GNN_drug(data)
        x_drug = self.drug_emb(x_drug)

        # cnv
        xcnv = self.cnv_ln(cnv)

        # rna
        xrna = self.rna_ln(rna)

        # met
        xmet = self.met_cnn(met)
        xmet = xmet.view(-1, xmet.shape[1] * xmet.shape[2])
        xmet = self.met_ln(xmet)

        # mut
        xmut = self.mut_cnn(mut)
        xmut = xmut.view(-1, xmut.shape[1] * xmut.shape[2])
        xmut = self.mut_ln(xmut)

        # fusion/prediction
        xfusion = torch.cat((x_drug, xcnv, xrna, xmet, xmut), 1)
        out = self.comb(xfusion)
        
        out = self.sigmoid(out)
        out = out.view(-1,1)
        return out

rmse = []
pcc = []
r2 = []
mape = []

LR = 1e-4

all_predictions = []
all_drug_names = []
all_cell_lines = []

for i in range(5):
    # Datasets
    with open(f'cross-val/train_fold_{i}.pkl', "rb") as f:
        train_dataset = pickle.load(f)
    with open(f'cross-val/validation_fold_{i}.pkl', 'rb') as f:
        val_dataset = pickle.load(f)
    with open(f'cross-val/test_fold_{i}.pkl', "rb") as f:
        test_dataset = pickle.load(f)

    train_loader = DataLoader(train_dataset,batch_size=1024,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=1024,shuffle=False)
    test_loader = DataLoader(test_dataset,batch_size=1024,shuffle=False)

    # Setting up model
    device = torch.device("cuda")
    model = DeepMoDRP().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    loss_function = nn.MSELoss()
    epochs = 300

    # Training and Validation
    best_mse = 100
    patience = 30
    counter = 0
    for epoch in range(epochs):

        # Train
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            prediction = model(batch)
            loss = loss_function(prediction,batch.ic50_targ)
            loss.backward()
            optimizer.step()
            
        # Validate 
        model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, batch.ic50_targ.cpu()), 0)

        mse = mean_squared_error(total_labels.numpy(),total_preds.numpy())

        if mse < best_mse:
            best_mse = mse
            counter = 0
            torch.save(model.state_dict(), f'model_fold_{i}.pt')
        else:
            counter += 1

        if epoch % 10 == 0:
            print(f'Fold: {i} | Epoch: {epoch + 1}/{epochs} | Validation MSE: {mse}')

        if counter >= patience:
            print(f'Early stopping at: {epoch+1}')
            break
    
    # Testing
    model.load_state_dict(torch.load(f'model_fold_{i}.pt'))
    model.eval()
    predictions = []
    actuals = []

    drug_names = []
    cell_lines = []

    with torch.no_grad():
        for batch in test_loader:
            drug_names.extend(batch.drug_name)
            cell_lines.extend(batch.cell_line) 

            batch = batch.to(device)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy())
            actuals.extend(batch.ic50_targ.cpu().numpy())

            

    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()

    all_predictions.extend(predictions)
    all_drug_names.extend(drug_names)
    all_cell_lines.extend(cell_lines)

    rmse.append(np.sqrt(mean_squared_error(actuals, predictions)))
    r2.append(r2_score(actuals, predictions))
    pcc_value, _ = pearsonr(actuals,predictions)
    pcc.append(pcc_value)
    mape.append(safe_mape(actuals, predictions))

    print(mape)

results = pd.DataFrame({
    "Drug": all_drug_names,
    "Cell_Line": all_cell_lines,
    "Predicted_IC50": all_predictions
})

results = results.sort_values(
    by="Predicted_IC50",
    ascending=True
)

top20 = results.head(int(len(results)*0.20))

top20.to_csv(
    "top20_predicted_drug_responses.csv",
    index=False
)

print(f'RMSE: {mean(rmse)}')
print(f'PCC: {mean(pcc)}')
print(f'R^2: {mean(r2)}')
print(f'MAPE: {mean(mape)}')