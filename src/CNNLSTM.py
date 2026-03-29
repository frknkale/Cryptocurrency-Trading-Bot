import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, target_len, params):
        super(CNNLSTM, self).__init__()
        self.output_dim = output_dim
        self.target_len = target_len
        self.epochs_trained = 0

        self.criterion = nn.MSELoss()

        # 1. Feature Extractor (Simplified to avoid overfitting on seq_len=14)
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2), # 14 -> 7

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(params.get('dropout', 0.2)),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool1d(4)

        # 2. LSTM
        num_layers = params.get('lstm_layers', 1)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=params.get('lstm_hidden_size', 32),
            num_layers=num_layers,
            batch_first=True,
            dropout=params.get('dropout', 0.2) if num_layers > 1 else 0
        )

        # 3. Output Head
        self.fc = nn.Sequential(
            nn.Linear(params.get('lstm_hidden_size', 32), 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(params.get('dropout', 0.2)),
            nn.Linear(32, self.target_len * self.output_dim)
        )

    def forward(self, x):
        # x shape: (Batch, Length, Channels) -> Permute to (Batch, Channels, Length)
        x = x.permute(0, 2, 1) 
        features = self.features(x)
        features = self.adaptive_pool(features)
        
        features = features.permute(0, 2, 1) 
        lstm_out, _ = self.lstm(features)
        
        # Take the last hidden state
        last_hidden_state = lstm_out[:, -1, :]
        
        prediction = self.fc(last_hidden_state)
        prediction = prediction.view(-1, self.target_len, self.output_dim)
        return prediction