"""
DFT code북 기반 빔포밍:
DFT 행렬로 서로 다른 방향을 가리키는 빔포밍 벡터들을 생성한다.
송신기는 각 코드북 빔으로 기준신호를 순차 전송하고,
UE는 각 빔의 수신 전력 |h^H w_k|^2을 측정하여 가장 강한 빔을 선택한다.
Steering vector는 이상적인 방향별 빔 패턴을 계산할 때 사용된다.
"""

import numpy as np
from DFT_codebook_generator import UPA_codebook_generator_DFT

N_antx = 4
N_anty = 1
N_antz = 1

codebook, _ = UPA_codebook_generator_DFT(
    Mx=N_antx, My=N_anty, Mz=N_antz)
