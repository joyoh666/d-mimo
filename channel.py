import torch
from sionna.phy.channel.tr38901 import TDL

def generate_dmimo_data(carrier_freq_hz):
    print(f"--- D-MIMO 시뮬레이션 시작 ({carrier_freq_hz/1e9} GHz) ---")
    
    num_bs = 3         # 분산된 기지국(RU) 개수
    num_tx_ant = 2     # 각 기지국당 안테나 개수
    num_ut = 1         # 단말(사용자) 개수
    num_rx_ant = 1     # 단말 안테나 개수
    num_samples = 100  # 추출할 시간 스텝 (5ms 간격 * 100 = 0.5초)

    h_list = []
    
    # 1. 3개의 기지국에 대해 각각 독립적인 채널 모델 생성
    for i in range(num_bs):
        tdl_model = TDL(model="A", 
                        delay_spread=30e-9, 
                        carrier_frequency=carrier_freq_hz, 
                        min_speed=10.0, 
                        max_speed=10.0,
                        num_rx_ant=num_rx_ant,
                        num_tx_ant=num_tx_ant)
        
        # [핵심 수정 부분] TimeChannel을 쓰지 않고 tdl_model에서 직접 h, tau를 추출합니다.
        # h_link 크기: [batch=1, rx=1, rx_ant=1, tx=1, tx_ant=2, time=100, path=23]
        h_link, tau_link = tdl_model(batch_size=1, num_time_steps=num_samples, sampling_frequency=1/0.005)
        h_list.append(h_link)
        
    # 2. 3개의 채널 텐서를 기지국(tx) 차원을 기준으로 병합 (Concatenation)
    # Sionna 텐서 차원: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, time, path]
    # 여기서 num_tx 차원(인덱스 3)을 기준으로 합칩니다.
    h_dmimo = torch.cat(h_list, dim=3)
    
    print("성공적으로 D-MIMO 채널을 생성했습니다!")
    print(f"최종 채널 텐서 크기: {h_dmimo.shape}")
    print("차원 해석: [Batch, UT개수, UT안테나, BS개수, BS안테나, 시간스텝, 다중경로]")
    
    return h_dmimo

if __name__ == "__main__":
    h_7_4ghz = generate_dmimo_data(7.4e9)