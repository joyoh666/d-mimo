import torch
import torch.nn as nn

class CSIPredictor(nn.Module):
    def __init__(self, csi_dim=15, age_dim=1, hidden_size=64):
        super(CSIPredictor, self).__init__()
        
        self.csi_dim = csi_dim
        self.age_dim = age_dim
        self.hidden_size = hidden_size
        
        # Input dimension: 15 (CSI features) + 1 (Age feature) = 16
        input_dim = self.csi_dim + self.age_dim
        
        # Single-layer GRU (Equation 10)
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            batch_first=True
        )
        
        # Output layer: \hat{x}[l+1] = W_o * u[l] + b_o (Equation 11)
        self.linear = nn.Linear(self.hidden_size, self.csi_dim)

    def forward(self, x, alpha, hidden=None):
        """
        Forward pass.
        :param x: Current CSI block (batch, seq, 15)
        :param alpha: Normalized age feature (batch, seq, 1)
        :param hidden: Previous hidden state u[l-1]
        """
        # Combine features: [x[l]; \alpha[l]]
        combined_input = torch.cat((x, alpha), dim=-1)
        
        # GRU operation updates the hidden state u[l]
        gru_out, hidden = self.gru(combined_input, hidden)
        
        # Linear layer outputs the prediction \hat{x}[l+1]
        prediction = self.linear(gru_out)
        
        return prediction, hidden