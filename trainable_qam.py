import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import sionna as sn
import sionna.phy

# 실험의 재현성을 위해 시드(Seed)를 고정합니다.
sionna.phy.config.seed = 42

# 1. 무작위 비트 생성기 준비
binary_source = sionna.phy.mapping.BinarySource()

# 2. 64-QAM 표준 성좌도 생성 (학습의 시작점/초기화 용도)
num_bits_per_symbol = 6
qam_constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol)

# 3. [핵심] 성좌도를 가중치(Weight)처럼 학습 가능한 파이토치 텐서로 변환
# 64개 복소수 점의 실수부(Real)와 허수부(Imag)를 추출해 2행 64열 형태로 만듭니다.
trainable_points = torch.stack([qam_constellation.points.real.clone(),
                                qam_constellation.points.imag.clone()], dim=0)

# 이 텐서의 미분 값(Gradient)을 계산하도록 설정하여 AI가 위치를 수정할 수 있게 만듭니다.
trainable_points.requires_grad_(True)

# 4. 미분 가능한 점들을 Sionna가 인식할 수 있는 'custom' 성좌도 객체로 등록
# normalize=True를 통해 점들이 이동하더라도 전체 평균 에너지가 1로 유지되도록 제약을 겁니다.
constellation = sionna.phy.mapping.Constellation("custom",
                                                 num_bits_per_symbol=num_bits_per_symbol,
                                                 points=torch.complex(trainable_points[0], trainable_points[1]),
                                                 normalize=True,
                                                 center=True)

# 5. 새롭게 정의된 (학습 가능한) 성좌도를 사용하는 매퍼와 디매퍼 선언
mapper = sionna.phy.mapping.Mapper(constellation=constellation)
demapper = sionna.phy.mapping.Demapper("app", constellation=constellation)

# 6. 통신 채널 및 환경 설정 (Eb/No = 17dB)
awgn_channel = sionna.phy.channel.AWGN()
batch_size = 128
ebno_db = 17.0

# 주어진 Eb/No에 맞는 잡음 전력(no) 계산
no = sionna.phy.utils.ebnodb2no(ebno_db=ebno_db,
                                num_bits_per_symbol=num_bits_per_symbol,
                                coderate=1.0)

# -------------------------------------------------------------------------
# [순전파 단계 (Forward Pass)] : 통신 데이터를 송수신합니다.
# -------------------------------------------------------------------------
# 128개 배치에 각각 1200비트씩 무작위 데이터 생성
bits = binary_source([batch_size, 1200])

x = mapper(bits)      # 비트를 우리가 만든 변형 가능한 성좌도 좌표(복소수)로 매핑
y = awgn_channel(x, no) # AWGN 채널을 통과시키며 잡음 주입
llr = demapper(y, no) # 수신기가 잡음 섞인 신호를 보고 각 비트의 확률(LLR)을 계산

# -------------------------------------------------------------------------
# [손실 계산 및 역전파 단계 (Loss & Backpropagation)] : 오차를 계산하고 수정 방향을 찾습니다.
# -------------------------------------------------------------------------
#Binary Cross Entropy 손실 함수를 사용해 수신기가 예측한 LLR과 원본 비트 사이의 오차를 계산합니다.
loss = F.binary_cross_entropy_with_logits(llr, bits.float())

# 오차를 바탕으로 역전파(Backpropagation)를 수행하여 미분 값을 계산합니다.
loss.backward()

# 'trainable_points'의 각 점들이 오차(Loss)를 줄이기 위해 어느 방향으로 움직여야 하는지 그라디언트를 확인합니다.
gradient = trainable_points.grad

# -------------------------------------------------------------------------
# [최적화 단계 (Optimization)] : 성좌도 점들의 위치를 실제로 이동시킵니다.
# -------------------------------------------------------------------------
# Adam 옵티마이저에 학습시킬 대상인 'trainable_points'를 등록합니다. (학습률 0.01)
optimizer = torch.optim.Adam([trainable_points], lr=1e-2)

# 가중치 업데이트 규칙에 따라 64개 성좌도 점들의 위치를 더 에러가 안 나는 방향으로 딱 '한 걸음' 이동시킵니다.
optimizer.step()

# -------------------------------------------------------------------------
# [결과 시각화 (Visualization)] : 바뀐 성좌도를 그래프로 그립니다.
# -------------------------------------------------------------------------
# 비교를 위해 원래 기본 64-QAM 성좌도 그래프를 먼저 배경에 띄웁니다.
fig = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol).show()

# 그 위에 딱 1번의 학습(GD)을 거쳐 미세하게 좌표가 이동한 새로운 점들의 위치를 주황색(After SGD) 점으로 덧그립니다.
# detach().numpy()를 해야 파이토치 추적에서 벗어나 matplotlib로 그리기가 가능해집니다.
fig.axes[0].scatter(trainable_points[0].detach().numpy(), 
                     trainable_points[1].detach().numpy(), 
                     label="After SGD", 
                     color="orange")
fig.axes[0].legend()

# 최종 그래프 출력
plt.show()