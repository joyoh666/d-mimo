import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os

# 모듈화된 모델 파일 불러오기
from GRU_wrapped import GRUwrapped
from GRU_unwrapped import GRUUnwrapped

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 공통 데이터 로더 및 학습 함수
# ==========================================
def load_dataset(file_path, batch_size=64):
    data = torch.load(file_path)
    X, Y = data['X'].to(device), data['Y'].to(device)
    
    tr_len = int(0.8 * len(X))
    train_ds = TensorDataset(X[:tr_len], Y[:tr_len])
    test_ds = TensorDataset(X[tr_len:], Y[tr_len:])
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), test_ds, data

def train(model, train_loader, epochs=100, K=5, eta=0.7):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    denom = sum(eta**i for i in range(K))
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            hidden = model.forward_context(batch_x)
            current_input = batch_x[:, -1, :] 
            
            batch_loss = 0.0
            for delta in range(K):
                alpha_val = (delta + 1) / K
                pred_x, hidden = model.forward_predict_step(current_input, alpha_val, hidden)
                step_mse = torch.mean((pred_x - batch_y[:, delta, :]) ** 2)
                batch_loss += (eta ** delta) * step_mse
                current_input = pred_x
                
            batch_loss = batch_loss / denom
            batch_loss.backward()
            optimizer.step()
            running_loss += batch_loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.6f}")
    return model

# ==========================================
# 2. 평가 함수 (각각 다름)
# ==========================================
def evaluate_wrapped(model, test_dataset, data_dict, K=5, target_delta=1):
    model.eval()
    evms_db = []
    gain_mean, gain_std = data_dict['gain_mean'].to(device), data_dict['gain_std'].to(device)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            X_seq, Y_seq = test_dataset[i]
            X_seq = X_seq.unsqueeze(0) 
            hidden = model.forward_context(X_seq)
            current_input = X_seq[:, -1, :]
            
            for delta in range(target_delta + 1): 
                pred_x, hidden = model.forward_predict_step(current_input, (delta + 1) / K, hidden)
                current_input = pred_x
            
            target_block = Y_seq[target_delta] 
            
            for s in range(5):
                pred_g, pred_c, pred_s = pred_x[0, s], pred_x[0, 5+s], pred_x[0, 10+s]
                true_g, true_c, true_s = target_block[s], target_block[5+s], target_block[10+s]
                
                # 정규화 해제 및 위상 단위원 보정 (기존 방식)
                pred_g_denorm = (pred_g * gain_std) + gain_mean
                true_g_denorm = (true_g * gain_std) + gain_mean
                
                phase_mag = torch.sqrt(pred_c**2 + pred_s**2) + 1e-8
                pred_c_norm, pred_s_norm = pred_c / phase_mag, pred_s / phase_mag
                
                pred_complex = pred_g_denorm * torch.complex(pred_c_norm, pred_s_norm)
                true_complex = true_g_denorm * torch.complex(true_c, true_s)
                
                error_p = torch.abs(pred_complex - true_complex) ** 2
                true_p = torch.abs(true_complex) ** 2
                
                if true_p > 1e-12:
                    evms_db.append(10 * np.log10((error_p / true_p).item()))
    return np.array(evms_db)

def evaluate_unwrapped(model, test_dataset, data_dict, K=5, target_delta=1):
    model.eval()
    evms_db = []
    gain_mean, gain_std = data_dict['gain_mean'].to(device), data_dict['gain_std'].to(device)
    phase_mean, phase_std = data_dict['phase_mean'].to(device), data_dict['phase_std'].to(device)
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            X_seq, Y_seq = test_dataset[i]
            X_seq = X_seq.unsqueeze(0) 
            hidden = model.forward_context(X_seq)
            current_input = X_seq[:, -1, :]
            
            for delta in range(target_delta + 1): 
                pred_x, hidden = model.forward_predict_step(current_input, (delta + 1) / K, hidden)
                current_input = pred_x
            
            target_block = Y_seq[target_delta] 
            
            for s in range(5):
                pred_g, pred_p = pred_x[0, s], pred_x[0, 5+s]
                true_g, true_p = target_block[s], target_block[5+s]
                
                # 정규화 해제 및 오일러 공식 자동 랩핑 (새로운 방식)
                pred_g_denorm = (pred_g * gain_std) + gain_mean
                true_g_denorm = (true_g * gain_std) + gain_mean
                
                pred_p_denorm = (pred_p * phase_std) + phase_mean
                true_p_denorm = (true_p * phase_std) + phase_mean
                
                pred_complex = pred_g_denorm * torch.exp(1j * pred_p_denorm)
                true_complex = true_g_denorm * torch.exp(1j * true_p_denorm)
                
                error_p = torch.abs(pred_complex - true_complex) ** 2
                true_p = torch.abs(true_complex) ** 2
                
                if true_p > 1e-12:
                    evms_db.append(10 * np.log10((error_p / true_p).item()))
    return np.array(evms_db)

# ==========================================
# 3. 메인 실행 및 시각화 (A/B 테스트)
# ==========================================
if __name__ == "__main__":
    print("🚀 [Step 1] 기존 모델(Wrapped, Cos/Sin) 학습 시작...")
    loader_wrap, test_wrap, dict_wrap = load_dataset("datasets/csi_7_4GHz_wrapped.pt")
    model_wrap = GRUwrapped().to(device)
    model_wrap = train(model_wrap, loader_wrap, epochs=100)
    evms_wrap = evaluate_wrapped(model_wrap, test_wrap, dict_wrap, target_delta=1)
    
    print("\n🚀 [Step 2] 새로운 모델(Unwrapped Phase) 학습 시작...")
    loader_unwrap, test_unwrap, dict_unwrap = load_dataset("datasets/csi_7_4GHz_unwrapped.pt")
    model_unwrap = GRUUnwrapped().to(device)
    model_unwrap = train(model_unwrap, loader_unwrap, epochs=100)
    evms_unwrap = evaluate_unwrapped(model_unwrap, test_unwrap, dict_unwrap, target_delta=1)

    print("\n📊 [Step 3] 결과 비교 그래프 생성 중...")
    plt.figure(figsize=(8, 6))
    
    for evm, label, color, style in [(evms_wrap, 'Wrapped (Cos/Sin)', 'blue', '--'), 
                                     (evms_unwrap, 'Unwrapped (Linear Phase)', 'red', '-')]:
        xs = np.sort(evm)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=label, color=color, linestyle=style, linewidth=2.5)
        
    plt.xlabel('EVM (dB)', fontsize=12)
    plt.ylabel('Empirical CDF', fontsize=12)
    plt.title('7.4 GHz Phase Prediction Comparison: Wrapped vs Unwrapped', fontsize=14)
    plt.xlim([-30, 10])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig("phase_comparison_result.png", dpi=300)
    print("🎉 'phase_comparison_result.png' 저장 완료! 완벽한 논문용 그래프가 생성되었습니다.")