import sionna.phy as sionna
import matplotlib.pyplot as plt
import numpy as np

batch_size = 1000       # numbers of symbol
num_bits_per_symbol = 4     # 16-QAM
binary_source = sionna.mapping.BinarySource()
bits = binary_source([batch_size, num_bits_per_symbol])     #shape :[1000, 4]

constellation = sionna.mapping.Constellation("qam", num_bits_per_symbol) #constellation

mapper = sionna.mapping.Mapper(constellation=constellation)        #map symbols to complex number
x = mapper(bits)

awgn = sionna.channel.AWGN()            #generate AWGN channel
ebn0_db = 15 # Desired Eb/N0 in dB
no = sionna.utils.ebnodb2no(ebn0_db, num_bits_per_symbol, coderate=1)   
#generate noise with ebno_dB (변조방식마다 심볼 에너지 다르므로 num_bits_per_symbol 필요)
y = awgn(x, no)

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
plt.scatter(np.real(y), np.imag(y))
ax.set_aspect("equal", adjustable="box")
plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.grid(True, which="both", axis="both")
plt.title("Received Symbols")
plt.show()