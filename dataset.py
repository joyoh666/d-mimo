import torch
import os
from sionna.phy.channel.tr38901 import TDL

def generate_data(carrier_freq_hz, num_samples=3000):
    print(f"\n[{carrier_freq_hz/1e9} GHz] 논문 True 스펙 블록 데이터 생성 시작...")
    
    total_tx_ants = 6
    tdl_model = TDL(model="A", delay_spread=30e-9, carrier_frequency=carrier_freq_hz,
                    min_speed=0.5, max_speed=0.5, num_rx_ant=1, num_tx_ant=total_tx_ants)
    
    sampling_freq = 1 / 0.005 # 5ms 간격
    h_complex, _ = tdl_model(batch_size=1, num_time_steps=num_samples, sampling_frequency=sampling_freq)
    
    L_p = 5 
    num_blocks = num_samples // L_p
    X_sequences, Y_sequences = [], []
    T_c = 10  
    K = 5     
    
    all_tokens = [] # 정규화를 위해 일단 모든 토큰을 모읍니다.

    for tx_idx in range(total_tx_ants):
        h_single_ant = h_complex[0, 0, 0, 0, tx_idx, 0, :]
        
        tokens = []
        for l in range(num_blocks):
            sample_start = l * L_p
            h_block = h_single_ant[sample_start : sample_start + L_p]
            gain = torch.abs(h_block)
            phase = torch.angle(h_block)
            x_l = torch.cat([gain, torch.cos(phase), torch.sin(phase)])
            tokens.append(x_l)
            
        all_tokens.append(torch.stack(tokens))

    # [중요] 전체 안테나의 토큰을 하나로 합친 후 Gain 부분(인덱스 0~4) 정규화
    all_tokens = torch.cat(all_tokens, dim=0) # Shape: [total_tx_ants * num_blocks, 15]
    
    gain_part = all_tokens[:, :5]
    gain_mean = gain_part.mean()
    gain_std = gain_part.std()
    
    # Gain 부분 표준화 적용
    all_tokens[:, :5] = (gain_part - gain_mean) / (gain_std + 1e-8)

    # 다시 안테나별로 나누어 슬라이딩 윈도우 시퀀스 구성
    all_tokens = all_tokens.view(total_tx_ants, num_blocks, 15)
    
    for tx_idx in range(total_tx_ants):
        tokens = all_tokens[tx_idx]
        for i in range(num_blocks - T_c - K + 1):
            X_sequences.append(tokens[i : i + T_c])
            Y_sequences.append(tokens[i + T_c : i + T_c + K])
            
    return torch.stack(X_sequences), torch.stack(Y_sequences), gain_mean, gain_std

if __name__ == "__main__":
    os.makedirs("datasets", exist_ok=True)
    
    X_22, Y_22, mean_22, std_22 = generate_data(2.2e9)
    torch.save({'X': X_22, 'Y': Y_22, 'gain_mean': mean_22, 'gain_std': std_22}, "datasets/csi_2_2GHz.pt")
    
    X_74, Y_74, mean_74, std_74 = generate_data(7.4e9)
    torch.save({'X': X_74, 'Y': Y_74, 'gain_mean': mean_74, 'gain_std': std_74}, "datasets/csi_7_4GHz.pt")
    
    print("\n🎉 논문 스펙 블록 데이터셋 생성 완료! (정규화 적용됨)")