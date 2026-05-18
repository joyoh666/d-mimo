import sionna.phy
import torch
import numpy as np
import matplotlib.pyplot as plt

sionna.phy.config.seed = 42

#Define the number of UT & BS antennas
num_UT = 1
num_BS = 1
num_UT_ant = 1
num_BS_ant = 4

#Assume the number of transmitted streams is equal to the number of
# UT antennas in both uplinlk and downlink
num_streams_per_TX = num_UT_ant

#Create a RX-TX association matrix
rx_tx_association = np.array([[1]])
stream_management = sionna.phy.mimo.StreamManagement(rx_tx_association, num_streams_per_TX)

#Resource Grid
resource_grid = sionna.phy.ofdm.ResourceGrid(num_ofdm_symbols=14,
                                             fft_size=76,
                                             subcarrier_spacing=30e3,
                                             num_tx=num_UT,
                                             num_streams_per_tx=num_streams_per_TX,
                                             cyclic_prefix_length=6,
                                             pilot_pattern="kronecker",
                                             pilot_ofdm_symbol_indices=[2, 11])
