import torch
import torch.nn as nn
import pandas as pd

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # 입력과 이전 hidden state를 합쳐서 입력한 다음 새로운 hidden state로 변환
        self.h2o = nn.Linear(hidden_size, output_size)               # hidden state를 출력으로 변환

    def forward(self, input, hidden):
        combined = torch.cat((input,hidden), 1)                      # 입력과 hidden state를 합쳐서 하나의 벡터로 변환
        hidden = torch.tanh(self.i2h(combined))
        output = self.h2o(hidden)
        return output, hidden
    
    def get_hidden(self):
        return torch.zeros(1, self.hidden_size)                     # 초기 hidden state를 0으로 설정하여 반환
    

rnn = MyRNN(input_size=4, hidden_size=4, output_size=2)
hidden = rnn.get_hidden()