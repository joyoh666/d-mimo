import sionna.phy
import matplotlib.pyplot as plt
import numpy as np

class AWGNsystem(sionna.phy.Block):
    def __init__(self, num_bits_per_symbol,block_length):
        """
        num_bits_per_symbol : constellation 심볼 당 비트 수
        block_length : 전송 메세지 블록 당 비트 수
        """
        super().__init__()

        self.num_bits_per_symbol = num_bits_per_symbol
        self.block_length = block_length
        self.constellation = sionna.phy.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.awgn_channel = sionna.phy.channel.AWGN()

    def call(self, batch_size, ebno_db):
        no = sionna.phy.utils.ebnodb2no(ebno_db,
                                        self.num_bits_per_symbol,
                                        coderate=1)
        bits = self.binary_source([batch_size, self.block_length])
        x = self.mapper(bits)
        y = self.awgn_channel(x, no)
        llr = self.demapper(y, no)
        return bits, llr
    
awgn = AWGNsystem(num_bits_per_symbol=4, block_length=1024)
ebno_db_min = -3.0
ebno_db_max = 5.0
batch_size = 2000

ber_plots = sionna.phy.utils.PlotBER("AWGN")
ber_plots.simulate(awgn,
                   ebno_dbs=np.linspace(ebno_db_min, ebno_db_max, 20),
                   batch_size=batch_size,
                   num_target_block_errors=100,
                   soft_estimates=True,
                   max_mc_iter=100,
                   show_fig=True)
plt.show()