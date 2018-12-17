import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=2):
            super().__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.batch_size = batch_size
            self.num_layers = num_layers
            
    def 

        