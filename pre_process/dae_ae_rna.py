import torch
from torch import nn
import csv
from torch.utils.data import DataLoader, TensorDataset
from sklearn import preprocessing
from scipy.stats import pearsonr
import pickle

L_R = 1e-4
BATCH_SIZE = 80
EPOCHS = 100

INPUT_SIZE = 17737
HIDDEN_SIZE = 256

def read_cell_line_dict(filename):
    f = open(filename, 'r')
    reader = csv.reader(f)
    next(reader)
    cell_line_dict = {}
    for i, line in enumerate(reader):
        cell_line_dict[line[0]] = i
    return cell_line_dict

def read_cell_line_cnv(filename, cell_line_dict):
    f = open(filename, 'r')
    reader = csv.reader(f)
    next(reader)
    cnv = [list() for _ in range(len(cell_line_dict))]
    for line in reader:
        if line[0] in cell_line_dict:
            cnv[cell_line_dict[line[0]]] = [float(x) if x != '' else 0.0 for x in line[1:]]
    return cnv

def load_data():
    cell_line_dict = read_cell_line_dict('datasets/cellline_listwithACH_80cellline.csv')
    cnv = read_cell_line_cnv('datasets/80cellline_17737dim_RNAseq.csv', cell_line_dict)
    
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1), copy=False)
    cnv = min_max_scaler.fit_transform(cnv)
    all_cnv_data = torch.FloatTensor(cnv)

    train_size = int(len(all_cnv_data)*0.8)
    train_data = all_cnv_data[:train_size]
    test_data = all_cnv_data[train_size:]

    return train_data, test_data, all_cnv_data

class sparse_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(          
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded
    
class denoising_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, hidden_size),
            nn.BatchNorm1d(hidden_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, input_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        noisy_x = x + 0.1 * torch.rand_like(x)
        x_encoded = self.encoder(noisy_x)
        x_decoded = self.decoder(x_encoded)
        return x_encoded, x_decoded
    
class combined_AE(nn.Module):
    def __init__(self, denoising, sparse):
        super().__init__()
        self.denoising_AE = denoising
        self.sparse_AE = sparse
        
    def forward(self, x):
        d_latent, d_rec = self.denoising_AE(x)
        s_latent, s_rec = self.sparse_AE(x)
        combined_latent = torch.cat([d_latent, s_latent], dim=1)
        combined_rec = d_rec + s_rec
        return combined_latent, combined_rec

def train(train_data, test_data, all_cnv_data):    
    train_loader = DataLoader(dataset= TensorDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)

    denoising = denoising_AE(INPUT_SIZE, HIDDEN_SIZE)
    sparse = sparse_AE(INPUT_SIZE, HIDDEN_SIZE)
    
    combined_model = combined_AE(denoising, sparse)
    loss_fn = nn.MSELoss()
    best_loss = float('inf')
    patience = 20
    counter = 0
    optimizer = torch.optim.Adam(combined_model.parameters(), lr=L_R)
    lat = [[0 for col in range(512)] for row in range(80)]

    for epoch in range(EPOCHS):
        for step , (data,) in enumerate(train_loader):
            train_latent, train_rec = combined_model(data)
            loss = loss_fn(train_rec, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        test_lat, test_rec = combined_model(test_data)
        test_loss = loss_fn(test_rec, test_data)
        if test_loss.item() < best_loss:
            best_loss = test_loss.item()
            counter = 0
            lat, _ = combined_model(all_cnv_data)
            pickle.dump(lat.data.numpy(), open('datasets/512dim_RNAseq.pkl', 'wb'))
            print(best_loss)
        else:
            counter += 1

        if counter >= patience:
            print(f"stopping at {epoch+1}")
            break

def ae_rna():
    train_data, test_data, all_rna_data = load_data()
    train(train_data, test_data, all_rna_data)
            


