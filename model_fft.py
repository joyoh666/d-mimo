import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DopplerSeq2Seq(nn.Module):
    def __init__(self, in_seq=10, out_seq=5, subcarriers=5):
        super(DopplerSeq2Seq, self).__init__()
        self.in_seq = in_seq
        self.out_seq = out_seq
        self.subcarriers = subcarriers

        # 입력: 과거 10스텝 * 5서브캐리어 * 2(Real/Imag) = 100차원
        in_features = in_seq * subcarriers * 2
        # 출력: 미래 5스텝 * 5서브캐리어 * 2(Real/Imag) = 50차원
        out_features = out_seq * subcarriers * 2
        
        # 주파수 도메인에서 피크(Peak)를 매핑해주는 핵심 신경망
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.GELU(), # 주파수 학습에 ReLU보다 조금 더 매끄러운 GELU 추천
            nn.Linear(128, out_features)
        )

    def forward(self, x_wrapped):
        # x_wrapped: [Batch, 10, 15] (Gain 5, Cos 5, Sin 5)
        batch_size = x_wrapped.shape[0]
        
        # 1. 🚨 입력 데이터를 완벽한 '복소수(Complex)'로 변환
        gain = x_wrapped[..., 0:5]
        cos_v = x_wrapped[..., 5:10]
        sin_v = x_wrapped[..., 10:15]
        
        phase_mag = torch.sqrt(cos_v**2 + sin_v**2) + 1e-8
        c_norm, s_norm = cos_v / phase_mag, sin_v / phase_mag
        x_complex = gain * torch.complex(c_norm, s_norm) # [Batch, 10, 5]
        
        # 2. 🚨 시간 축 -> 주파수(Doppler) 축으로 FFT 변환
        x_fft = torch.fft.fft(x_complex, dim=1) 
        
        # 3. 주파수 도메인 신경망 연산을 위해 Real/Imag 분리 및 평탄화(Flatten)
        x_real = x_fft.real.reshape(batch_size, -1)
        x_imag = x_fft.imag.reshape(batch_size, -1)
        x_in = torch.cat([x_real, x_imag], dim=-1) # [Batch, 100]
        
        # 4. MLP 통과 (과거 도플러 피크 -> 미래 도플러 피크 매핑)
        out = self.mlp(x_in) # [Batch, 50]
        
        # 5. 출력을 다시 주파수 도메인의 복소수 형태로 조립
        out_real, out_imag = torch.split(out, out.shape[-1] // 2, dim=-1)
        out_real = out_real.reshape(batch_size, self.out_seq, self.subcarriers)
        out_imag = out_imag.reshape(batch_size, self.out_seq, self.subcarriers)
        out_fft = torch.complex(out_real, out_imag) # [Batch, 5, 5]
        
        # 6. 🚨 주파수 축 -> 미래 시간 축으로 IFFT (역 푸리에 변환)
        pred_time_complex = torch.fft.ifft(out_fft, dim=1) 
        
        # 이 모델은 최종적으로 [Batch, 5, 5] 형태의 완벽한 '미래 복소수 채널'을 뱉어냅니다!
        return pred_time_complex