import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import config
from CNNLSTM import CNNLSTM
from data.csv_data import read_csv


def forecast_model(
    input_combinations: list, 
    model_name: str = "LSTM", 
    forecast_type: str = "log_ret_close", 
    coin: str = "BTC", 
    time_frame: str = "1d"
):
    
    all_needed_cols = sorted(list(set(sum(input_combinations, []))))
    if forecast_type not in all_needed_cols:
        all_needed_cols.append(forecast_type)
        
    df = read_csv(coin, time_frame, all_needed_cols).dropna()
    
    seq_len = 14
    target_len = 1
    
    split_index = int(len(df) * (1 - config.test_percentage))
    split_date = df.index[split_index]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for feature_combo in tqdm(input_combinations, desc=f"Testing {coin} Feature Combos"):
        
        input_name = "__".join(feature_combo)
        save_dir = f"{config.output_path}/model_predictions/{model_name}/input_{input_name}/{forecast_type}/{coin}/{time_frame}"
        os.makedirs(save_dir, exist_ok=True)
        
        target_idx = feature_combo.index(forecast_type)
        data_values = df[feature_combo].values
        
        X, Y, dates = [], [], []
        for i in range(len(data_values) - seq_len):
            X.append(data_values[i : i + seq_len, :])       
            Y.append(data_values[i + seq_len, target_idx])  
            dates.append(df.index[i + seq_len])             
            
        X_np = np.array(X)
        Y_np = np.array(Y)
        dates_array = np.array(dates)
        
        train_mask = dates_array < split_date
        test_mask = ~train_mask
        
        X_train_np, Y_train_np = X_np[train_mask], Y_np[train_mask]
        X_test_np, Y_test_np = X_np[test_mask], Y_np[test_mask]
        test_dates = dates_array[test_mask]
        
        # Z-Score Scaling (Fitted ONLY on training data to avoid leakage)
        for c in range(X_train_np.shape[2]):
            col_mean = X_train_np[:, :, c].mean()
            col_std = X_train_np[:, :, c].std() + 1e-8 # add epsilon to avoid div by zero
            
            X_train_np[:, :, c] = (X_train_np[:, :, c] - col_mean) / col_std
            X_test_np[:, :, c]  = (X_test_np[:, :, c] - col_mean) / col_std
            
        # Convert back to tensors
        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        Y_train = torch.tensor(Y_train_np, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        
        X_test = torch.tensor(X_test_np, dtype=torch.float32)
        Y_test = torch.tensor(Y_test_np, dtype=torch.float32).unsqueeze(1).unsqueeze(2)
        
        train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=16, shuffle=True)
        
        params = {'lstm_hidden_size': 32, 'lstm_layers': 1, 'dropout': 0.3}
        
        model = CNNLSTM(
            input_dim=len(feature_combo), 
            output_dim=1, 
            target_len=target_len, 
            params=params
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.MSELoss()gi
        
        model.train()
        epochs = 25
        for epoch in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
        model.eval()
        with torch.no_grad():
            preds = model(X_test.to(device))
            predictions = preds.view(-1).cpu().numpy()
            
        df_pred = pd.DataFrame({"date": test_dates, forecast_type: predictions})
        df_pred.to_csv(f"{save_dir}/pred.csv", index=False)
        
        df_actual = pd.DataFrame({"date": test_dates, forecast_type: Y_test.view(-1).numpy()})
        df_actual.to_csv(f"{save_dir}/actual.csv", index=False)
        
        with open(f"{save_dir}/features.txt", "w") as f:
            f.write(str(feature_combo))