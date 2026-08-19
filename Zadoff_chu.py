import numpy as np
import matplotlib.pyplot as plt

"""
Zadoff-Chu sequence 
일정한 크기의 진폭을 가지며, 신호 간섭이 적은 복소수 형태의 수학적 수열로, 채널 추정 및 동기화에 사용.
자기상관 특성: 시간 차이가 나도 자기 자신과의 상관이 낮음
상호상관 특성: 다른 Zadoff-Chu 수열과의 상관이 낮음
"""
def zadoff_chu_seq(R, N):
    seq = np.zeros(N, dtype=complex)
    for n in range(N):
        seq[n] = np.exp(-1j * np.pi * R * n * (n + 1) / N)
    return seq

def selfcorrelation(seq):
    N = len(seq)
    corr = np.zeros(N, dtype=complex)
    for k in range(N):
        corr[k] = np.sum(seq * np.roll(np.conj(seq), k))
    return corr/N

def crosscorrelation(seq1, seq2):
    N = len(seq1)
    corr = np.zeros(N, dtype=complex)
    for k in range(N):
        corr[k] = np.sum(seq1 * np.roll(np.conj(seq2), k))
    return corr/N

seq1 = zadoff_chu_seq(1, 63)
seq2 = zadoff_chu_seq(2, 63)

plt.plot(np.abs(crosscorrelation(seq1, seq2)))
plt.show()