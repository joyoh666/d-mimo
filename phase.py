import torch
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 및 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

def visualize_dataset_phase(file_path="datasets/csi_7_4GHz.pt", sample_idx=0):
    print(f"📦 데이터 로드 중: {file_path}")
    data = torch.load(file_path)
    
    X, Y = data['X'], data['Y']
    gain_mean, gain_std = data['gain_mean'], data['gain_std']
    
    # 1. 과거(T_c=10) + 미래(K=5) = 총 15개 블록 추출
    full_seq = torch.cat([X[sample_idx], Y[sample_idx]], dim=0) # [15, 15]
    
    # 2. 블록(L_p=5) 구조를 원래의 시간 스텝으로 쫙 펴주기 (Flatten)
    # full_seq[:, :5] 는 [15, 5] 크기이며, 이를 flatten하면 75개의 연속된 샘플이 됩니다.
    gain_norm = full_seq[:, :5].flatten()
    cos_v = full_seq[:, 5:10].flatten()
    sin_v = full_seq[:, 10:15].flatten()
    
    # 3. Gain 역정규화 (원래의 물리적 크기 복원)
    gain_denorm = (gain_norm * gain_std) + gain_mean
    
    # 4. 복소수 신호 복원 및 시간 축 위상 계산
    # 생성 코드에서 샘플링 간격을 1/0.005 (5ms)로 설정했으므로 간격은 0.005초입니다.
    sampling_interval = 0.005 
    time_steps = np.arange(len(gain_denorm)) * sampling_interval * 1000 # ms 단위로 변환
    
    complex_sig = gain_denorm * (cos_v + 1j * sin_v)
    complex_sig = complex_sig.cpu().numpy()
    
    time_phase_rad = np.angle(complex_sig)
    time_phase_deg = np.degrees(time_phase_rad)
    
    # 5. 주파수 축 (FFT) 변환
    fft_result = np.fft.fft(complex_sig)
    fft_freqs = np.fft.fftfreq(len(complex_sig), d=sampling_interval)
    
    fft_mag = np.abs(fft_result)
    fft_phase_rad = np.angle(fft_result)
    fft_phase_deg = np.degrees(fft_phase_rad)

    # ==========================================
    # 🎨 그래프 시각화 (A/B 비교)
    # ==========================================
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 📉 [그래프 1] 시간 축에서의 위상 (Time Domain)
    axes[0].plot(time_steps, time_phase_deg, marker='o', color='red', linestyle='-', linewidth=1.5, markersize=4)
    axes[0].set_title("1. 시간 축 위상 변화 (7.4GHz, 0.5m/s)", fontsize=14)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("phase (Degrees)")
    axes[0].set_ylim([-180, 180])
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].axhline(0, color='black', linewidth=0.8)

    # 주파수 데이터 정렬 (보기 편하게 중앙을 0Hz로 맞춤)
    freqs_shifted = np.fft.fftshift(fft_freqs)
    mag_shifted = np.fft.fftshift(fft_mag)
    phase_shifted = np.fft.fftshift(fft_phase_deg)
    
    # 메인 피크(가장 강한 도플러 성분) 찾기
    max_idx = np.argmax(mag_shifted)
    main_freq = freqs_shifted[max_idx]
    main_mag = mag_shifted[max_idx]
    main_phase = phase_shifted[max_idx]

    # 📊 [그래프 2] 주파수 축에서의 크기 (Doppler Spectrum)
    markerline, stemlines, baseline = axes[1].stem(freqs_shifted, mag_shifted)
    plt.setp(markerline, color='blue', markersize=6)
    plt.setp(stemlines, color='blue', linewidth=2)
    axes[1].set_title("2. 주파수 축 크기 (도플러 스펙트럼)", fontsize=14)
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("magnitude (Magnitude)")
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].annotate(f'Main Peak: {main_freq:.1f} Hz', 
                     xy=(main_freq, main_mag), xytext=(main_freq+5, main_mag),
                     arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12)

    # 🎯 [그래프 3] 주파수 축에서의 위상 (Phase Spectrum)
    markerline, stemlines, baseline = axes[2].stem(freqs_shifted, phase_shifted)
    plt.setp(markerline, color='green', markersize=6)
    plt.setp(stemlines, color='green', linewidth=2)
    axes[2].set_title("3. 주파수 축 위상 (메인 피크의 위상 고정 확인)", fontsize=14)
    axes[2].set_xlabel("frequency (Hz)")
    axes[2].set_ylabel("phase (Degrees)")
    axes[2].set_ylim([-180, 180])
    axes[2].grid(True, linestyle='--', alpha=0.7)
    
    # 메인 피크 위치의 위상 하이라이트
    axes[2].scatter([main_freq], [main_phase], color='magenta', s=150, zorder=5)
    axes[2].annotate(f'고정된 위상: {main_phase:.1f}°', 
                     xy=(main_freq, main_phase), xytext=(main_freq+5, main_phase+30),
                     arrowprops=dict(facecolor='magenta', shrink=0.05), fontsize=12, color='magenta')

    plt.tight_layout()
    plt.savefig("my_dataset_phase_analysis.png", dpi=300)
    print("🎉 'my_dataset_phase_analysis.png' 저장 완료! 데이터셋 분석이 성공적으로 끝났습니다.")
    plt.show()

if __name__ == "__main__":
    visualize_dataset_phase("datasets/csi_7_4GHz.pt", sample_idx=10) # 10번째 샘플 확인 (변경 가능)