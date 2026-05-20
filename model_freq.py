import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqDomainCNN(nn.Module):
    def __init__(self, num_antennas=5, hidden_channels=128, K=5):
        super(FreqDomainCNN, self).__init__()
        self.num_antennas = num_antennas
        self.K = K  # 한 번에 예측할 미래 블록의 수 (One-shot Prediction)
        
        # 복소수 채널을 Real/Imag 2개로 쪼개서 넣으므로 입력 채널은 안테나 수의 2배 (10채널)
        in_channels = num_antennas * 2
        
        # 주파수 도메인에서 글로벌 패턴을 읽어낼 1D-CNN
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, in_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, gain_mean, gain_std):
        # x shape: [Batch, T_c, 15]
        B, T_c, _ = x.shape
        
        g = x[:, :, 0:5]
        c = x[:, :, 5:10]
        s = x[:, :, 10:15]
        
        # 1. 정규화 해제 및 실제 물리적 복소수 채널(H) 구성
        g_denorm = (g * gain_std) + gain_mean
        H = g_denorm * torch.complex(c, s)  # shape: [B, T_c, 5]
        
        # 2. 미래 스텝을 담을 공간을 위해 시간축(dim=1)을 뒤로 K만큼 Zero-padding
        # padding=(마지막 차원 앞, 뒤, 두번째 차원 앞, 뒤)
        H_padded = F.pad(H, (0, 0, 0, self.K))  # shape: [B, T_c + K, 5]
        
        # 3. 시간 도메인 -> 주파수 도메인 (FFT)
        H_freq = torch.fft.fft(H_padded, dim=1)  # shape: [B, T_c + K, 5]
        
        # 4. 1D-CNN 처리를 위해 데이터 형태 변환 (Real/Imag 분리 후 채널 결합)
        H_freq_real = H_freq.real.transpose(1, 2)  # [B, 5, T_c + K]
        H_freq_imag = H_freq.imag.transpose(1, 2)  
        cnn_in = torch.cat([H_freq_real, H_freq_imag], dim=1)  # [B, 10, T_c + K]
        
        # 5. 주파수 스펙트럼 매핑 (CNN이 주파수 도메인에서의 위상 회전 규칙을 학습)
        cnn_out = self.cnn(cnn_in)  # [B, 10, T_c + K]
        
        # 6. 다시 복소수로 결합 후 차원 복구
        out_real = cnn_out[:, 0:5, :]
        out_imag = cnn_out[:, 5:10, :]
        H_freq_pred = torch.complex(out_real, out_imag).transpose(1, 2)  # [B, T_c + K, 5]
        
        # 7. 주파수 도메인 -> 시간 도메인 복원 (IFFT)
        H_time_pred = torch.fft.ifft(H_freq_pred, dim=1)  # [B, T_c + K, 5]
        
        # 8. 우리가 예측하려던 미래 K스텝만 잘라내기
        H_future = H_time_pred[:, -self.K:, :]  # [B, K, 5]
        
        # 9. 원본 형태(G, C, S)로 분리 및 Gain 재정규화 (기존 학습/평가 코드와 호환을 위해)
        pred_g_denorm = torch.abs(H_future)
        pred_g = (pred_g_denorm - gain_mean) / gain_std
        
        pred_phase = H_future / (pred_g_denorm + 1e-8)
        pred_c = pred_phase.real
        pred_s = pred_phase.imag
        
        # 최종 출력 shape: [Batch, K, 15]
        pred_x = torch.cat([pred_g, pred_c, pred_s], dim=-1)
        
        return pred_x