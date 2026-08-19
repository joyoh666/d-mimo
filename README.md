# D-MIMO Prototype OFDM Sounding Simulation

The primary simulation in this folder models the manuscript's single-UE
sounding and CSI-estimation chain in complex baseband:

1. transmitter.py creates a Zadoff-Chu synchronization preamble and six
   simultaneous frequency-orthogonal pilot streams.
2. channel.py applies one short multipath channel per transmit branch,
   sums the branches at the single receive antenna, and adds timing offset,
   carrier-frequency offset (CFO), and AWGN.
3. receiver.py performs ZC frame synchronization, repeated-pilot CFO correction,
   CP removal, FFT, per-branch LS channel estimation, and complex averaging
   over three consecutive pilot symbols.
4. main.py runs the complete chain and saves a synchronization/CSI figure.
5. test_chain.py verifies the waveform parameters and end-to-end recovery.

## Manuscript parameters represented directly

- Center frequency: 2.2 GHz
- Nominal bandwidth: 1.4 MHz
- Subcarrier spacing: 15 kHz
- FFT size: 128
- Complex-baseband sample rate: 1.92 MS/s (128 * 15 kHz)
- Six transmit branches: three RUs with two antennas each
- One receive branch
- CSI acquisition period: 5 ms
- Three consecutive pilots averaged for each CSI estimate
- One distinct pilot subcarrier per transmit branch

The center frequency is metadata in this baseband simulation; an actual RF
transmission at 2.2 GHz requires UHD and USRP hardware.

## Explicit simulation assumptions

The manuscript does not specify the exact CP length, ZC root/length, guard
length, or the six pilot-bin indices. config.py therefore exposes these as
replaceable assumptions. The default channel taps, 250-sample timing offset,
120 Hz CFO, and SNR are also simulation settings rather than measured values.

Run the demonstration and standalone tests with:

    .venv/bin/python main.py
    .venv/bin/python test_chain.py

The demonstration writes output/sounding_demo.png.

---

# Fig. 2 reproduction: Spectral Efficiency of Distributed MIMO Systems

`fig2_reproduction.py` reproduces Fig. 2 of D. Wang et al., "Spectral
Efficiency of Distributed MIMO Systems," IEEE JSAC, vol. 31, no. 10, 2013.

The implementation uses the paper's stated parameters:

- D-MIMO dimensions: `(M, L, N) = (2, 2, 7)`
- Cell-edge SNR: `gamma = 20 dB`
- User angle: `theta = 0 degrees`
- Normalized user radius: `rho/D = 0.02, 0.04, ..., 1.0`
- Path-loss exponent: `alpha = 3.7`
- Lognormal shadowing standard deviation: `sigma_sh = 8 dB`
- Ring RAU radius: `(3 - sqrt(3))D/2`

The four plotted curves are:

1. Exact Monte Carlo evaluation of equation (11)
2. Lower bound from equation (30)
3. High-SNR asymptotic approximation from equation (36)
4. Schwartz-Yeh lognormal-sum approximation applied to equation (35)

Run with the existing project environment:

```bash
.venv/bin/python fig2_reproduction.py
```

For a quicker smoke test:

```bash
.venv/bin/python fig2_reproduction.py --trials 10000
```

The default outputs are:

- `output/figures/fig2_reproduction.png`
- `output/figures/fig2_reproduction.csv`

Use `--show` to open the Matplotlib window, or `--help` for all options.
