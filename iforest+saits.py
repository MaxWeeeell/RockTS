import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES']='6'

parser = argparse.ArgumentParser()
parser.add_argument('--anom_data', type=str, default='ETTh2', help='data name')
args = parser.parse_args()


file_path = 'data/'+args.anom_data+'_anom.csv'  
df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')


class SAITS(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads):
        super(SAITS, self).__init__()
        self.embedding = nn.Linear(20, hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 20)

    def forward(self, x):
        # x = x.unsqueeze(-1)
        # print(x.shape)
        x= x.reshape(-1, 5, 20)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        x = x.reshape(-1,100)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


for col in df.columns:

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[[col]].dropna())


    iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    preds = iso_forest.fit_predict(scaled_data)


    anomaly_mask = preds == -1


    data_tensor = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(1).to(device)
    print(int(data_tensor.shape[0]/100) * 100)


    input_dim = 1
    hidden_dim = 128
    num_layers = 2
    num_heads = 1
    model = SAITS(input_dim, hidden_dim, num_layers, num_heads).to(device)


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    num_epochs = 100
    data_tensor = data_tensor[:int(data_tensor.shape[0]/100) * 100]
    data_tensor = data_tensor.reshape(-1,100)
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(data_tensor)
        # print(output)
        loss = criterion(output, data_tensor)
        loss.backward()
        optimizer.step()
        print(f'Column: {col}, Epoch {epoch+1}, Loss: {loss.item()}')


    model.eval()
    with torch.no_grad():
        imputed_data = model(data_tensor).to('cpu').squeeze(1).numpy()
        imputed_data = imputed_data.reshape(-1)

    imputed_data = scaler.inverse_transform(imputed_data.reshape(-1, 1))


    pad_length = len(df)-imputed_data.shape[0]
    imputed_data = np.pad(imputed_data, pad_width=((0, pad_length), (0, 0)), mode='constant', constant_values=0)
    df.loc[anomaly_mask, col] = imputed_data[anomaly_mask,0]



cleaned_file_path = 'data/'+args.anom_data+'_clean.csv' 
df.to_csv(cleaned_file_path, index=True)