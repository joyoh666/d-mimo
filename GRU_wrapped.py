import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUwrapped(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=15):
        super(GRUwrapped, self).__init__()
        # 입력 차원을 input_size로 유지 (외부에서 15+1=16으로 맞춤)
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # T_c 개의 블록 sequence를 받아서 hidden state 반환(예측값이 아닌 실제 값만을 이용)
    def forward_context(self, context_x):
        batch_size, T_c, _ = context_x.shape
        alpha_zero = torch.zeros(batch_size, T_c, 1).to(device)
        gru_in = torch.cat([context_x, alpha_zero], dim=-1)
        _, hidden = self.gru(gru_in)
        return hidden

    # 과거 예측값에서 추출한 hidden state, 현재 입력값, alpha_value를 받아서 다음 블록 예측
    def forward_predict_step(self, prev_x, alpha_value, hidden):
        batch_size = prev_x.shape[0]
        alpha_tensor = torch.full((batch_size, 1, 1), alpha_value).to(device)
        gru_in = torch.cat([prev_x.unsqueeze(1), alpha_tensor], dim=-1)
        out, hidden = self.gru(gru_in, hidden)
        pred_x = self.fc(out.squeeze(1))
        return pred_x, hidden