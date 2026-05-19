import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUUnwrapped(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, output_size=10):
        super(GRUUnwrapped, self).__init__()
        # input: 10(Gain 5 + Phase 5) + 1(Alpha) = 11
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward_context(self, context_x):
        batch_size, T_c, _ = context_x.shape
        alpha_zero = torch.zeros(batch_size, T_c, 1).to(device)
        gru_in = torch.cat([context_x, alpha_zero], dim=-1)
        _, hidden = self.gru(gru_in)
        return hidden

    def forward_predict_step(self, prev_x, alpha_value, hidden):
        batch_size = prev_x.shape[0]
        alpha_tensor = torch.full((batch_size, 1, 1), alpha_value).to(device)
        gru_in = torch.cat([prev_x.unsqueeze(1), alpha_tensor], dim=-1)
        out, hidden = self.gru(gru_in, hidden)
        pred_x = self.fc(out.squeeze(1))
        return pred_x, hidden