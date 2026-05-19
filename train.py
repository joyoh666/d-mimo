import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AutoregressiveGRU(nn.Module):
    def __init__(self, input_size=16, hidden_size=64, output_size=15):
        super(AutoregressiveGRU, self).__init__()
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
    
def load_dataset(file_path, batch_size=64):
    data = torch.load(file_path)
    X, Y = data['X'].to(device), data['Y'].to(device)
    
    # 🚨 정규화 파라미터 로드 (없을 경우 기본값 0, 1)
    gain_mean = data.get('gain_mean', torch.tensor(0.0)).to(device)
    gain_std = data.get('gain_std', torch.tensor(1.0)).to(device)
    
    tr_len = int(0.8 * len(X))
    
    # 🚨 시계열 데이터 누수(Leakage) 방지를 위해 random_split 대신 순차적 분할 사용
    train_ds = TensorDataset(X[:tr_len], Y[:tr_len])
    test_ds = TensorDataset(X[tr_len:], Y[tr_len:])
    
    # 학습 데이터로더만 셔플 적용
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), test_ds, gain_mean, gain_std

def train(model, train_loader, epochs=100, K=5, eta=0.7):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    denom = sum(eta**i for i in range(K))

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            # T_c개의 블록 시퀀스를 보고 hidden state 추출
            hidden = model.forward_context(batch_x)
            current_input = batch_x[:, -1, :] 
            
            batch_loss = 0.0
            for delta in range(K):
                alpha_val = (delta + 1) / K
                # hidden state와 현재 입력값, alpha_val을 이용해 바로 다음 블록을 예측 -> 예측값을 다시 입력값으로 넣어서 다음 블록 예측 반복
                pred_x, hidden = model.forward_predict_step(current_input, alpha_val, hidden)
                
                step_mse = torch.mean((pred_x - batch_y[:, delta, :]) ** 2)
                batch_loss += (eta ** delta) * step_mse
                current_input = pred_x
                
            batch_loss = batch_loss / denom
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Weighted Loss: {running_loss/len(train_loader):.6f}")
    return model

def evaluate(model, test_dataset, gain_mean, gain_std, K=5, target_delta=1):
    model.eval()
    evms_db = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            X_seq, Y_seq = test_dataset[i]
            X_seq = X_seq.unsqueeze(0) 
            
            hidden = model.forward_context(X_seq)
            current_input = X_seq[:, -1, :]
            
            # target_delta(기본 1 -> T+2 블록)까지 예측 스텝 진행
            for delta in range(target_delta + 1): 
                alpha_val = (delta + 1) / K
                pred_x, hidden = model.forward_predict_step(current_input, alpha_val, hidden)
                current_input = pred_x
            
            target_block = Y_seq[target_delta] 
            
            for s in range(5):
                pred_g, pred_c, pred_s = pred_x[0, s], pred_x[0, 5+s], pred_x[0, 10+s]
                true_g, true_c, true_s = target_block[s], target_block[5+s], target_block[10+s]
                
                # 1. Gain 정규화 해제 (Denormalization)
                pred_g_denorm = (pred_g * gain_std) + gain_mean
                true_g_denorm = (true_g * gain_std) + gain_mean
                
                # Phase 단위원 보정 (Normalization)
                phase_mag = torch.sqrt(pred_c**2 + pred_s**2) + 1e-8
                pred_c_norm = pred_c / phase_mag
                pred_s_norm = pred_s / phase_mag
                
                # 3. 안정적인 복소수 복원 (보정된 위상 사용!)
                pred_complex = pred_g_denorm * torch.complex(pred_c_norm, pred_s_norm)
                true_complex = true_g_denorm * torch.complex(true_c, true_s) # 실제 정답(true)은 이미 길이가 1이므로 보정 불필요
                
                error_p = torch.abs(pred_complex - true_complex) ** 2
                true_p = torch.abs(true_complex) ** 2
                
                # 신호 전력이 0에 가까운 비정상 케이스 방어
                if true_p > 1e-12:
                    evm_linear = (error_p / true_p).item() 
                    evms_db.append(10 * np.log10(evm_linear))
                    
    return np.array(evms_db)

if __name__ == "__main__":
    # 데이터셋 존재 여부 확인
    if not os.path.exists("datasets/csi_2_2GHz.pt") or not os.path.exists("datasets/csi_7_4GHz.pt"):
        print("🚨 데이터셋이 존재하지 않습니다. 먼저 generate_data 코드를 실행해주세요.")
        exit()

    # 2.2 GHz 대역
    print("--- 2.2 GHz 대역 학습 ---")
    loader_2, test_2, mean_2, std_2 = load_dataset("datasets/csi_2_2GHz.pt")
    model_2 = AutoregressiveGRU().to(device)
    model_2 = train(model_2, loader_2)
    # T+2(delta=1) 미래 예측 EVM 측정
    evms_2 = evaluate(model_2, test_2, mean_2, std_2, K=5, target_delta=1)
    
    # 7.4 GHz 대역
    print("\n--- 7.4 GHz 대역 학습 ---")
    loader_7, test_7, mean_7, std_7 = load_dataset("datasets/csi_7_4GHz.pt")
    model_7 = AutoregressiveGRU().to(device)
    model_7 = train(model_7, loader_7)
    # T+2(delta=1) 미래 예측 EVM 측정
    evms_7 = evaluate(model_7, test_7, mean_7, std_7, K=5, target_delta=1)
    
    # CDF 플롯 출력
    plt.figure(figsize=(7, 5))
    for evm, label, color, style in [(evms_2, '2.2 GHz', 'blue', '-'), (evms_7, '7.4 GHz', 'red', '--')]:
        xs = np.sort(evm)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=label, color=color, linestyle=style, linewidth=2)
        
    plt.xlabel('EVM (dB)')
    plt.ylabel('Empirical CDF')
    plt.title('True Paper Spec AR-GRU Prediction (Denormalized)')
    plt.xlim([-30, 10])
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("true_paper_result_corrected.png", dpi=300)
    print("\n🎉 'true_paper_result_corrected.png' 그래프 저장 완료!")