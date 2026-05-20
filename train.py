import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
from model_freq import FreqDomainCNN
from model_time import AutoregressiveGRU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

def train_time(model, train_loader, epochs=100, K=5, eta=0.7):
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

def evaluate_time(model, test_dataset, gain_mean, gain_std, K=5, target_delta=1):
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

def train_freq(model, train_loader, gain_mean, gain_std, epochs=100, K=5, eta=0.7):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    denom = sum(eta**i for i in range(K))

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            
            # 모델이 K개의 미래 블록을 한 번에 예측 (One-shot)
            pred_y = model(batch_x, gain_mean, gain_std)  # pred_y shape: [B, K, 15]
            
            # 시간 가중치(eta)를 적용한 Loss 계산 (기존 아이디어 유지)
            batch_loss = 0.0
            for delta in range(K):
                step_mse = torch.mean((pred_y[:, delta, :] - batch_y[:, delta, :]) ** 2)
                batch_loss += (eta ** delta) * step_mse
                
            batch_loss = batch_loss / denom
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Weighted Loss: {running_loss/len(train_loader):.6f}")
    return model

def evaluate_freq(model, test_dataset, gain_mean, gain_std, target_delta=1):
    model.eval()
    evms_db = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            X_seq, Y_seq = test_dataset[i]
            X_seq = X_seq.unsqueeze(0)  # [1, T_c, 15]
            
            # 한 번의 통과로 K개 스텝 전체 예측
            pred_y_all = model(X_seq, gain_mean, gain_std)
            
            # 원하는 미래 시점의 블록 추출 (delta=1 이면 T+2 블록)
            pred_block = pred_y_all[0, target_delta, :]
            target_block = Y_seq[target_delta] 
            
            for s in range(5):
                pred_g, pred_c, pred_s = pred_block[s], pred_block[5+s], pred_block[10+s]
                true_g, true_c, true_s = target_block[s], target_block[5+s], target_block[10+s]
                
                # 1. Gain 정규화 해제 (Denormalization)
                pred_g_denorm = (pred_g * gain_std) + gain_mean
                true_g_denorm = (true_g * gain_std) + gain_mean
                
                # 2. Phase 단위원 보정 (Normalization)
                phase_mag = torch.sqrt(pred_c**2 + pred_s**2) + 1e-8
                pred_c_norm = pred_c / phase_mag
                pred_s_norm = pred_s / phase_mag
                
                # 3. 복소수 복원
                pred_complex = pred_g_denorm * torch.complex(pred_c_norm, pred_s_norm)
                true_complex = true_g_denorm * torch.complex(true_c, true_s) 
                
                error_p = torch.abs(pred_complex - true_complex) ** 2
                true_p = torch.abs(true_complex) ** 2
                
                if true_p > 1e-12:
                    evm_linear = (error_p / true_p).item() 
                    evms_db.append(10 * np.log10(evm_linear))
                    
    return np.array(evms_db)

if __name__ == "__main__":
    # 데이터셋 존재 여부 확인
    if not os.path.exists("datasets/csi_7_4GHz.pt"):
        print("🚨 데이터셋이 존재하지 않습니다. 먼저 generate_data 코드를 실행해주세요.")
        exit()
    
    # time domain
    print("\n--- Time domain 학습 ---")
    loader, test, mean, std = load_dataset("datasets/csi_7_4GHz.pt")
    model_t = AutoregressiveGRU().to(device)
    model_t = train_time(model_t, loader)
    # T+2(delta=1) 미래 예측 EVM 측정
    evms_t = evaluate_time(model_t, test, mean, std, K=5, target_delta=1)

    #frequency domain
    print("\n--- Frequency domain 학습 ---")
    model_f = FreqDomainCNN(K=5).to(device)
    model_f = train_freq(model_f, loader, mean, std)
    evms_f = evaluate_freq(model_f, test, mean, std, target_delta=1)
    
    # CDF 플롯 출력
    plt.figure(figsize=(7, 5))
    for evm, label, color, style in [(evms_t, 'Time domain', 'red', '--'), (evms_f, 'Frequency Domain', 'blue', '-')]:
        xs = np.sort(evm)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=label, color=color, linestyle=style, linewidth=2)
        
    plt.xlabel('EVM (dB)')
    plt.ylabel('Empirical CDF')
    plt.title('Time domain vs Frequency domain')
    plt.xlim([-30, 10])
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.savefig("Time domain vs Frequency domain.png", dpi=300)
    print("\n🎉 'Time domain vs Frequency domain.png' 그래프 저장 완료!")