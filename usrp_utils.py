import numpy as np
import uhd
from threading import Thread

# usrp_utils version 2025-05-20 18:50

def sendAndReceive(usrp, signal, power, Fc, Fs,
                   Tx_gain=0, Rx_gain=10, Tx_chan=None, Rx_chan=None,
                   wait_time=0.2, tx_delay_samples=100, rx_trailing_samples=100,
                   otw_format='sc16'):
    '''
    usrp: usrp object.
    signal: (np.array, dtype=complex64) numpy array of complex-valued baseband signal.
    power: (float) constant value to be multiplied in signal.
    Fc: (float) carrier frequency
    Fs: (float) sampling rate
    Tx_gain: (float/list) Tx gain (in dB) for each channels. Typically 0 - 31.5
             If float, the value is broadcasted for all channels.
    Rx_gain: (float/list) Rx gain (in dB) for each channels. Typically 0 - 31.5
             If float, the value is broadcasted for all channels.
    Tx_chan: (list) Tx channels
    Rx_chan: (list) Rx channels
    wait_time: (float) time delay for initial sample reception (waiting for tx/rx thread spawn)
    tx_delay_samples: (int) time delay for initial sample transmission (avoiding large noise in initial rx samples)
    rx_trailing_samples: (int) additional samples received (accounting for tx timing error)
    otw_format: ('sc16' or 'sc8') symbol representation type within connection of PC and USRP.
                'sc16' uses 16-bit complex integer values and provides better resolution,
                while 'sc8' uses 8-bit representations with doubled Msps (samples per second) between USRP and PC links.
                'sc8' not supported in X310.
    '''

    # Timing control
    current_time = usrp.get_time_now().get_real_secs()
    rx_time = current_time +  wait_time
    tx_time = rx_time # + tx_delay_samples / Fs

    # Tx/Rx channel/gain sanity check
    if Rx_chan is None:
        Rx_chan = list(range(usrp.get_rx_num_channels()))

    if Tx_chan is None:
        Tx_chan = list(range(usrp.get_tx_num_channels()))
    
    if isinstance(Tx_gain, (int, float, complex)) and not isinstance(Tx_gain, bool):
        # If Tx_gain is constant:
        Tx_gain = [Tx_gain] * len(Tx_chan)

    if isinstance(Rx_gain, (int, float, complex)) and not isinstance(Rx_gain, bool):
        # If Rx_gain is constant:
        Rx_gain = [Rx_gain] * len(Rx_chan)
        
    # Rx
    rx_thread = USRPReceiver(
        usrp=usrp,
        num_samples=int(signal.shape[-1] + tx_delay_samples + rx_trailing_samples),
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Rx_gain,
        channels=Rx_chan,
        transmission_time=rx_time,
        otw_format=otw_format
    )

    # Tx
    signal *= np.sqrt(power)

    zero_pad = np.zeros((signal.shape[0], tx_delay_samples), dtype=np.complex64)
    signal = np.concatenate([zero_pad, signal], axis=1)

    tx_thread = USRPTransmitter(
        usrp=usrp,
        samples=signal.astype(np.complex64),
        carrier_frequency=Fc,
        sampling_rate=Fs, 
        gain=Tx_gain,
        channels=Tx_chan,
        transmission_time=tx_time,
        otw_format=otw_format
    )

    rx_thread.start()
    tx_thread.start()
    tx_thread.join()
    rx_thread.join()
    
    rcv_signal = rx_thread.rcv_samples
    rcv_signal /= np.sqrt(power)
    
    return rcv_signal

class USRPReceiver(Thread):
    def __init__(self, usrp, num_samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, otw_format):
        super().__init__()
        self.usrp = usrp
        self.num_samples = num_samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.otw_format = otw_format
        
        for i, c in enumerate(channels):
          usrp.set_rx_rate(self.sampling_rate, c)
          usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
          usrp.set_rx_gain(self.gain[i], c)
          usrp.set_rx_dc_offset(False, c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", otw_format) # possibly 'sc8' for more data rate instead of more quantization error
        st_args.channels = channels
        self.streamer = self.usrp.get_rx_stream(st_args)
        self.num_samples_per_frame = self.streamer.get_max_num_samps()
    

    def receiveUSRP(self, num_samples, metadata):
        num_channels = len(self.channels)

        recv_buffer = np.zeros((num_channels, self.num_samples_per_frame), dtype=np.complex64)
        samples = np.zeros((num_channels, num_samples), dtype=np.complex64)

        head, tail = 0, 0
        try:
            while tail < num_samples:
                tail += self.streamer.recv(recv_buffer, metadata)
                tail = min(tail, num_samples)
                samples[:, head:tail] = recv_buffer[:, :tail-head]
                head = tail

                if metadata.error_code != uhd.types.RXMetadataErrorCode.none:
                    # print(metadata.error_code)
                    if metadata.error_code != uhd.types.RXMetadataErrorCode.timeout:
                        break
        except RuntimeError as ex:
            print(ex)
        
        return samples
    
    
    def run(self):
        # Start Stream
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done) # num_done, stop_cont, start_cont
        stream_cmd.stream_now = False
        stream_cmd.time_spec = uhd.types.TimeSpec(self.transmission_time)
        stream_cmd.num_samps = self.num_samples
        self.streamer.issue_stream_cmd(stream_cmd)
        
        metadata = uhd.types.RXMetadata()

        self.rcv_samples = self.receiveUSRP(self.num_samples, metadata)
        
        # Close streamer
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
        self.streamer.issue_stream_cmd(stream_cmd)
        self.streamer = None
        
        
class USRPTransmitter(Thread):
    def __init__(self, usrp, samples, carrier_frequency, sampling_rate, gain, channels, transmission_time, otw_format):
        super().__init__()
        self.usrp = usrp
        assert samples.shape[0] == len(channels), "sample.shape[0] != # channel"
        self.samples = samples
        self.carrier_frequency = carrier_frequency
        self.sampling_rate = sampling_rate
        self.gain = gain
        self.channels = channels
        self.transmission_time = transmission_time
        self.otw_format = otw_format
        
        for i, c in enumerate(channels):
            self.usrp.set_tx_rate(self.sampling_rate, c)
            self.usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(self.carrier_frequency), c)
            self.usrp.set_tx_gain(self.gain[i], c)
        
        # Set up the stream and receive buffer
        st_args = uhd.usrp.StreamArgs("fc32", otw_format)
        st_args.channels = channels
        self.streamer = self.usrp.get_tx_stream(st_args)
        
    
    def run(self):
        # Transmit Samples
        metadata = uhd.types.TXMetadata()
        metadata.has_time_spec = True
        metadata.time_spec = uhd.types.TimeSpec(self.transmission_time)

        try:
            i = self.streamer.send(self.samples, metadata)
            while i < self.samples.shape[1]:
                metadata.has_time_spec = False
                i += self.streamer.send(self.samples[:, i:], metadata)
        except RuntimeError as ex:
            print(ex)
        
        # End
        metadata.end_of_burst = True
        self.streamer.send(np.zeros((self.samples.shape[0], 0), dtype=np.complex64), metadata)
        self.streamer = None
