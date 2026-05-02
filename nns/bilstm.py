import torch.nn as nn
import torch

class BILSTM(nn.Module):
    def __init__(self, args, embedding_dim=768, bilstm_dim=128, encoding=False):
        super().__init__()          
        self.encoding = encoding
        self.lstm = nn.LSTM(embedding_dim,
                            bilstm_dim,
                            dropout=args.dropout,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        
    def forward(self, embedded):
        h_lstm, (hidden, _) = self.lstm(embedded)
        if self.encoding:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            return hidden
            """
            avg_pool = torch.mean(h_lstm, 1)
            max_pool, _ = torch.max(h_lstm, 1)
            conc = torch.cat((avg_pool, max_pool), 1)
            return conc
            """
        else:
            return h_lstm
        
