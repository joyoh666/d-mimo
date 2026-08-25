#%%
import sys
import os
# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
import uhd
import matplotlib.pyplot as plt
from usrp_utils import sendAndReceive

from IPython import display
# %matplotlib widget

import matplotlib.style as mplstyle
mplstyle.use(['dark_background', 'fast'])

from USRP.modulate import modulate, modulations

carrier_frequency = 2.2e9
Tx_gain, Rx_gain = 15, 30

# Number of subcarriers
N, FFT_SIZE = (72, 128)

# Note: (72, 128) for 1.4MHz, (180, 256) for 3MHz, (300, 512) for 5MHz
# (600, 1024) for 10 / (900, 1536) for 15 / (1200, 2048) for 20

delta_f = 15e3  # Subcarrier spacing in Hz
CP_time = 4.7e-6 # Normal CP: 4.7us
modulation_order = 16
N_symbol_per_frame = 70 # 1 frame = 70 symbols (CSI sampling interval : 5 ms)
N_PSS = N
POWER = 1/8

num_symbols = N * int(np.log2(modulation_order))
T = 1 / delta_f # OFDM symbol time without CP in seconds
sampling_rate = delta_f * FFT_SIZE
CP_length = int((delta_f * FFT_SIZE) * CP_time)
num_pad_pss_front = FFT_SIZE - (N_PSS+1) - ((FFT_SIZE - (N_PSS+1))//2)
num_pad_pss_end = (FFT_SIZE - (N_PSS+1))//2
bb_length = (CP_length+FFT_SIZE)*N_symbol_per_frame

num_RU = 3
num_antennaperRU = 2
num_channel = num_RU * num_antennaperRU

#%%
# USRP preparation =========================================================
#서로 다른 USRP 장비를 multiUSRP 코드로 묶어서 MIMO의 역할 수행
usrp = uhd.usrp.MultiUSRP('addr0=192.168.110.2, addr1=192.168.10.2, addr2=192.168.100.2')

#%%
#USRP의 주파수 기준과 시간 기준을 octoclock에서 온 외부 신호를 사용하도록 설정
for i in range(num_RU):
    usrp.set_clock_source("external",i)
    usrp.set_time_source("external",i)

# 외부의 시간 시준 신호(PPS)를 받았을 때 내부 시간을 0으로 설정
usrp.set_time_unknown_pps(uhd.types.TimeSpec(0.0))
print("USRP loaded. Session Ready.")

#USRP 내부에서 어떤 RF frontend를 tx channel로 사용하고 선택된 RF frontend의 신호를 어느 
#RF 커넥터로 출력할지 결정
# RU 1 : X310 #1
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),0)
# RU 2 : X310 #2
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),1)
# RU 3 : X310 #3
usrp.set_tx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),2)

# RU 1 : X310 #1
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),0)
# RU 2 : X310 #2
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),1)
# RU 3 : X310 #3
usrp.set_rx_subdev_spec(uhd.usrp.SubdevSpec("A:0 B:0"),2)

print('Tx:')
print(usrp.get_tx_subdev_spec(0))
print(usrp.get_tx_subdev_spec(1))
print('Rx:')
print(usrp.get_rx_subdev_spec(0))
print(usrp.get_rx_subdev_spec(1))

#%%
# OFDM preparation =========================================================
def get_zc_sequence(N, q=25):
    m = np.arange(N)
    seq = -np.pi * q * m * (m+1) / N

    i = np.cos(seq)
    q = np.sin(seq)
    return i + 1j*q

# Add PSS (ZC sequence, 62 subcarrier) in the front of the frame
pss_zc = get_zc_sequence(N_PSS)
dc = np.zeros((1), dtype=np.complex64)
pss = np.concatenate([
    dc, # DC
    pss_zc[N_PSS//2:], # right half to positive frequency
    np.zeros(num_pad_pss_front, dtype=np.complex64),
    np.zeros(num_pad_pss_end, dtype=np.complex64),
    pss_zc[:N_PSS//2], # left half to negative frequency
], axis=-1)
pss = np.reshape(pss, (1, FFT_SIZE))
pss_td = np.fft.ifft(pss, FFT_SIZE).astype(np.complex64) * np.sqrt(N)
pss_td_with_cp = np.concatenate([pss_td[:, -CP_length:], pss_td], axis=-1)
pss_td_with_cp = np.asarray(pss_td_with_cp.flatten())

#%%

# Tx Signal Processing
iq_pad = np.zeros((N_symbol_per_frame - 1, FFT_SIZE), dtype=np.complex64)
ofdm_symbol_with_cp = np.zeros((N_symbol_per_frame, CP_length + FFT_SIZE), dtype=np.complex64)

input_bits = np.random.randint(0, 2, num_symbols * (N_symbol_per_frame - 1), dtype=np.uint8)
iq = modulate(input_bits, modulation_order)

iq = np.reshape(iq, (N_symbol_per_frame - 1, N))
# (n_symbol_per_frame, #_subcarriers)

# insert zero in the middle (do not use DC area)
iq_pad[:, 1:N//2+1] = iq[:, N//2:] # right half is put in the positive area
iq_pad[:, -(N//2):] = iq[:, :N//2] # left half is put in the negative area

ofdm_td = np.fft.ifft(iq_pad, FFT_SIZE).astype(np.complex64)

# Add PSS
ofdm_td_scaled = ofdm_td * np.sqrt(N)
ofdm_symbol_with_cp[1:, CP_length:] = ofdm_td_scaled
ofdm_symbol_with_cp[0, CP_length:] = pss_td
ofdm_symbol_with_cp[:, :CP_length] = ofdm_symbol_with_cp[:, -CP_length:]
frame = ofdm_symbol_with_cp.flatten() * np.sqrt(POWER)

# Guard
NUM_GUARD_SAMPLES = 1000
i = np.linspace(0, 10*np.pi, NUM_GUARD_SAMPLES)
waveform_sin = (np.sin(i) + 1j*np.cos(i)).astype(np.complex64)

waveform = np.zeros((num_channel, bb_length + NUM_GUARD_SAMPLES), dtype=np.complex64)
for ch in range(num_channel):
    waveform[ch] = np.concatenate([waveform_sin, frame], axis=0) # Single Antenna

#%%
# USRP send
frame_rcv = sendAndReceive(usrp, waveform, 1, carrier_frequency, sampling_rate, Tx_gain, Rx_gain, [0], [0],
    wait_time=0.2, tx_delay_samples=20, rx_trailing_samples=200, otw_format='sc16')

#%%
# Rx Signal Processing
frame_rcv = frame_rcv[0].flatten() # Single Antenna

# corr = np.abs(scipy.signal.correlate(frame_rcv, pss_td_with_cp.flatten(), 'valid', 'fft'))
corr = np.abs(np.correlate(np.asarray(frame_rcv), pss_td_with_cp, 'valid'))
corr_weight = 1 - np.arange(corr.size) / corr.size # linear decay
sync_idx = np.argmax(corr * corr_weight)
    
if sync_idx + bb_length > frame_rcv.size:
    sync_idx = 0

frame_rcv_sync = frame_rcv[sync_idx:sync_idx+bb_length]

ofdm_td_rcv = np.reshape(frame_rcv_sync, (N_symbol_per_frame, -1))

# Channel estimation
pss_rcv_raw = np.fft.fft(ofdm_td_rcv[0, CP_length:]).astype(np.complex64)

pss_rcv = np.zeros(N_PSS, dtype=np.complex64)
pss_rcv[:N_PSS//2] = pss_rcv_raw[-(N_PSS-N_PSS//2):] # negative freq = left half
pss_rcv[N_PSS//2:] = pss_rcv_raw[1:N_PSS//2+1] # positive freq = right half

h = pss_rcv / pss_zc # shape: (N_PSS)

# Currently N == N_PSS, so ignore the following code
# # extend size of h from PSS (N_PSS) to N
# h = np.concatenate([
#     np.repeat(h[0], N//2-N_PSS//2),
#     h,
#     np.repeat(h[-1], (N - N_PSS)-(N//2-N_PSS//2))
# ])

h = np.reshape(h, (1, -1))

ofdm_symbol = np.fft.fft(ofdm_td_rcv[1:, CP_length:]).astype(np.complex64)
iq_rcv = np.zeros((N_symbol_per_frame-1, N), dtype=np.complex64)
iq_rcv[:, :N//2] = ofdm_symbol[:, -(N-N//2):] # negative freq = left half
iq_rcv[:, N//2:] = ofdm_symbol[:, 1:N//2+1] # positive freq = right half

iq_rcv = iq_rcv / h
iq_rcv = iq_rcv.flatten()

#%%
# Plot =========================================================
fig, axs = plt.subplots(ncols=4, nrows=5, figsize=(8, 10))
gs1 = axs[0, 0].get_gridspec()
gs2 = axs[0, 2].get_gridspec()
gs3 = axs[1, 0].get_gridspec()
for ax in axs.flatten():
    ax.remove()
ax1 = fig.add_subplot(gs1[0, 0:2])
ax2 = fig.add_subplot(gs2[0, 2:4])
ax3 = fig.add_subplot(gs3[1:, :])
fig.tight_layout()
dh = display.display(fig, display_id=True)

subcarrier_idx = np.arange(N)

# frame_rcv, iq_rcv, h
ax1.plot(np.abs(h.flatten()))
ax1.set_xlim([0, N])
ax1.set_xlabel('Subcarrier Index')
ax1.set_ylabel('Channel Magnitude')

ax2.psd(frame_rcv, NFFT=frame_rcv.size, Fs=sampling_rate, scale_by_freq=False)
ax2.set_xlim([-sampling_rate//2, sampling_rate//2])
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Power Spectrum (dB)')

real_gt = np.real(modulations[modulation_order])
imag_gt = np.imag(modulations[modulation_order])
peak_amp = 2 * np.max(real_gt)

ax3.scatter(np.real(iq_rcv), np.imag(iq_rcv), s=0.5, marker='o')
ax3.scatter(real_gt, imag_gt, s=10.0, marker='o')
ax3.set_xlim([-peak_amp, peak_amp])
ax3.set_ylim([-peak_amp, peak_amp])
ax3.set_xlabel('In-Phase')
ax3.set_ylabel('Quadrature-Phase')

dh.update(fig)
plt.close()

# %%
