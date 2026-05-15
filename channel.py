import torch
import sionna
from sionna.phy.channel import TimeChannel
from sionna.phy.channel.tr38901 import CDL, AntennaArray
from sionna.phy.mimo import StreamManagement

def generate_csi_data(carrier_freq_hz, velocity_mps, num_time_steps, batch_size=1):
    """
    Sionna 2.0 (PyTorch 기반)을 이용해 
    시계열 복소 채널(CSI) 데이터를 생성합니다.
    """
    
    # 1. 안테나 설정 (단순 정수가 아닌 AntennaArray 객체 필요)
    # 기지국(BS): 2개 안테나, 단말(UT): 1개 안테나 예시
    ut_array = AntennaArray(num_rows=1, num_cols=1, polarization="single", 
                            antenna_pattern="38.901", carrier_frequency=carrier_freq_hz)
    bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="single", 
                            antenna_pattern="38.901", carrier_frequency=carrier_freq_hz)

    # 2. 스트림 관리
    # rx_tx_association: [num_rx, num_tx] 형태의 boolean tensor
    rx_tx_association = torch.tensor([[True]]) 
    # num_rx_ant 대신 num_streams_per_tx 사용
    stream_management = StreamManagement(rx_tx_association, num_streams_per_tx=1)

    # 3. CDL 채널 모델 설정
    cdl = CDL(model="A",
              delay_spread=30e-9, 
              carrier_frequency=carrier_freq_hz, 
              ut_array=ut_array,
              bs_array=bs_array,
              direction="downlink", # "forward" 대신 "downlink" 또는 "uplink"
              min_speed=velocity_mps, 
              max_speed=velocity_mps) 

    # 4. 시간 도메인 채널 생성기
    # sampling_frequency는 대역폭(Bandwidth)과 관련이 있습니다.
    sampling_frequency = 20e6 # 예: 20MHz
    time_channel = TimeChannel(cdl, 
                               bandwidth=sampling_frequency, 
                               num_time_samples=num_time_steps, 
                               l_min=-5, l_max=5) 

    # 5. 채널 데이터 생성
    # h: [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, num_l]
    h, tau = time_channel(batch_size=batch_size)
    
    return h, tau

if __name__ == "__main__":
    velocity = 10.0 # 10 m/s
    num_steps = 100 # 측정 포인트 개수
    
    # 2.2 GHz 대역
    h_2_2ghz, _ = generate_csi_data(2.2e9, velocity, num_steps)
    
    # 7.4 GHz 대역
    h_7_4ghz, _ = generate_csi_data(7.4e9, velocity, num_steps)

    print(f"2.2 GHz 채널 텐서 형태: {h_2_2ghz.shape}")
    print(f"7.4 GHz 채널 텐서 형태: {h_7_4ghz.shape}")