# ================================================================
# Slot 55 - Beamforming and Codebook-Based Precoding Simulator
#                    for 5G NR
#
# Technology: Python
# Simulation Type: 5G NR / mmWave Beamforming
#
# Main Deliverables:
# 1. Beamforming Simulator
# 2. DFT Beam Patterns
# 3. Codebook-Based Beam Selection
# 4. Beam Gain vs UE Angle
# 5. SINR vs UE Angle
# 6. UE Mobility and Beam Tracking
# 7. Selection Accuracy Analysis
# ================================================================


# ------------------------------------------------
# 1. IMPORT REQUIRED LIBRARIES
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from USRP.DFTsweeping.DFT_codebook_generator import UPA_codebook_generator_DFT

# ------------------------------------------------
# 2. CREATE PROJECT RESULT FOLDER
# ------------------------------------------------

# The Python file is inside:
# 5G-WIPRO-Beamforming-Codebook-Simulator/src/
#
# parents[1] refers to:
# 5G-WIPRO-Beamforming-Codebook-Simulator/

PROJECT_DIR = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_DIR / "results"

# Create results folder automatically if it doesn't exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# 3. SIMULATION PARAMETERS
# ------------------------------------------------

# Number of antenna elements
N = 8

# Number of predefined beams
NUM_BEAMS = 8

# Carrier frequency = 28 GHz
fc = 28e9

# Speed of light
c = 3e8

# Calculate wavelength
wavelength = c / fc

# Half-wavelength antenna spacing
d = wavelength / 2

# Noise power
noise_power = 0.01

# Interference power
interference_power = 0.10

# Random seed makes the simulation repeatable
np.random.seed(10)


# ------------------------------------------------
# 4. PRINT BASIC SIMULATION INFORMATION
# ------------------------------------------------

print("=" * 65)
print("5G NR BEAMFORMING AND CODEBOOK-BASED PRECODING SIMULATOR")
print("=" * 65)

print(f"Number of antenna elements : {N}")
print(f"Number of DFT beams        : {NUM_BEAMS}")
print(f"Carrier frequency          : {fc / 1e9:.1f} GHz")
print(f"Wavelength                 : {wavelength:.6f} m")
print(f"Antenna spacing            : {d:.6f} m")
print(f"Noise power                : {noise_power}")
print(f"Interference power         : {interference_power}")
print("=" * 65)


# ------------------------------------------------
# 5. STEERING VECTOR FUNCTION
# ------------------------------------------------

def steering_vector(theta_deg, N):
    """
    Generate the normalized steering vector of an
    N-element Uniform Linear Array for a given angle.

    Parameters
    ----------
    theta_deg : float
        UE angle in degrees.

    N : int
        Number of antenna elements.

    Returns
    -------
    numpy.ndarray
        Normalized complex steering vector.
    """

    # Convert angle from degrees to radians
    theta_rad = np.deg2rad(theta_deg)

    # Antenna element indices
    n = np.arange(N)

    # Calculate the complex steering vector
    a = np.exp(1j * np.pi * n * np.sin(theta_rad))

    # Normalize the vector
    a = a / np.sqrt(N)

    return a


# ------------------------------------------------
# 6. GENERATE DFT BEAM CODEBOOK
# ------------------------------------------------

# Generate the 8-beam DFT codebook
codebook, _ = UPA_codebook_generator_DFT(
    Mx=N, My=1, Mz=1)


print("\nDFT codebook generated successfully.")
print(f"Codebook shape: {codebook.shape}")
print("Each row represents one predefined beam.")


# ------------------------------------------------
# 7. CALCULATE BEAM GAIN
# ------------------------------------------------

def calculate_beam_gain(beam, channel):
    """
    Calculate received beam gain between a beamforming
    vector and the channel steering vector.
    """

    gain = np.abs(np.vdot(beam, channel)) ** 2

    return gain


# ------------------------------------------------
# 8. CREATE DFT BEAM PATTERNS
# ------------------------------------------------

# Angular range used for beam pattern visualization
pattern_angles = np.linspace(-90, 90, 721)


plt.figure(figsize=(10, 6))

for k in range(NUM_BEAMS):

    beam_pattern = []

    for angle in pattern_angles:

        # Channel/steering vector for current angle
        a = steering_vector(angle, N)

        # Calculate beam gain
        gain = calculate_beam_gain(codebook[k], a)

        beam_pattern.append(gain)

    beam_pattern = np.array(beam_pattern)

    # Normalize to the maximum value
    beam_pattern_normalized = (
        beam_pattern / np.max(beam_pattern)
    )

    # Convert to dB
    beam_pattern_db = 10 * np.log10(
        beam_pattern_normalized + 1e-10
    )

    # Plot beam
    plt.plot(
        pattern_angles,
        beam_pattern_db,
        label=f"Beam {k}"
    )


plt.title("DFT Beam Patterns for 8-Element Antenna Array")
plt.xlabel("Angle (Degrees)")
plt.ylabel("Normalized Beam Gain (dB)")
plt.grid(True)
plt.legend()
plt.ylim(-40, 2)
plt.tight_layout()

# Save graph
beam_pattern_file = RESULTS_DIR / "dft_beam_patterns.png"
plt.savefig(beam_pattern_file, dpi=300)

plt.show()


print("\nBeam pattern graph generated.")
print(f"Saved to: {beam_pattern_file}")


# ------------------------------------------------
# 9. BEAM GAIN AND BEST BEAM VS UE ANGLE
# ------------------------------------------------

# UE angular positions
ue_angles = np.linspace(-60, 60, 241)

max_gains = []
selected_beams = []


for ue_angle in ue_angles:

    # Create channel for current UE direction
    h = steering_vector(ue_angle, N)

    beam_gains = []

    # Test every codebook beam
    for k in range(NUM_BEAMS):

        gain = calculate_beam_gain(
            codebook[k],
            h
        )

        beam_gains.append(gain)

    beam_gains = np.array(beam_gains)

    # Select the beam with maximum gain
    best_beam = np.argmax(beam_gains)

    # Store best beam
    selected_beams.append(best_beam)

    # Store maximum gain
    max_gains.append(
        beam_gains[best_beam]
    )


max_gains = np.array(max_gains)
selected_beams = np.array(selected_beams)


# ------------------------------------------------
# 10. CONVERT BEAM GAIN TO dB
# ------------------------------------------------

max_gains_db = 10 * np.log10(
    max_gains + 1e-10
)


# ------------------------------------------------
# 11. PLOT BEAM GAIN VS UE ANGLE
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    ue_angles,
    max_gains_db,
    linewidth=2
)

plt.title("Beam Gain vs UE Angle")
plt.xlabel("UE Angle (Degrees)")
plt.ylabel("Maximum Beam Gain (dB)")
plt.grid(True)
plt.tight_layout()

beam_gain_file = RESULTS_DIR / "beam_gain_vs_ue_angle.png"

plt.savefig(
    beam_gain_file,
    dpi=300
)

plt.show()

print("\nBeam gain analysis completed.")
print(f"Saved to: {beam_gain_file}")


# ------------------------------------------------
# 12. PLOT CODEBOOK-BASED BEAM SELECTION
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.step(
    ue_angles,
    selected_beams,
    where="mid",
    linewidth=2
)

plt.title("Codebook-Based Beam Selection vs UE Angle")
plt.xlabel("UE Angle (Degrees)")
plt.ylabel("Selected Beam Index")
plt.yticks(range(NUM_BEAMS))
plt.grid(True)
plt.tight_layout()

beam_selection_file = (
    RESULTS_DIR /
    "codebook_beam_selection_vs_ue_angle.png"
)

plt.savefig(
    beam_selection_file,
    dpi=300
)

plt.show()

print("\nCodebook beam selection analysis completed.")
print(f"Saved to: {beam_selection_file}")


# ------------------------------------------------
# 13. SINR CALCULATION
# ------------------------------------------------

def calculate_sinr(signal_power,
                    interference_power,
                    noise_power):
    """
    Calculate SINR and return the result in dB.
    """

    sinr_linear = (
        signal_power /
        (interference_power + noise_power)
    )

    sinr_db = 10 * np.log10(
        sinr_linear + 1e-10
    )

    return sinr_db


# ------------------------------------------------
# 14. CALCULATE SINR VS UE ANGLE
# ------------------------------------------------

sinr_values = []


for ue_angle in ue_angles:

    # Channel for current UE angle
    h = steering_vector(ue_angle, N)

    beam_gains = []

    # Evaluate all codebook beams
    for k in range(NUM_BEAMS):

        gain = calculate_beam_gain(
            codebook[k],
            h
        )

        beam_gains.append(gain)

    beam_gains = np.array(beam_gains)

    # Select strongest beam
    best_beam = np.argmax(
        beam_gains
    )

    # Desired signal power
    signal_power = beam_gains[best_beam]

    # Calculate SINR
    sinr_db = calculate_sinr(
        signal_power,
        interference_power,
        noise_power
    )

    sinr_values.append(sinr_db)


sinr_values = np.array(sinr_values)


# ------------------------------------------------
# 15. PLOT SINR VS UE ANGLE
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    ue_angles,
    sinr_values,
    linewidth=2
)

plt.title("SINR vs UE Angle")
plt.xlabel("UE Angle (Degrees)")
plt.ylabel("SINR (dB)")
plt.grid(True)
plt.tight_layout()

sinr_file = RESULTS_DIR / "sinr_vs_ue_angle.png"

plt.savefig(
    sinr_file,
    dpi=300
)

plt.show()

print("\nSINR analysis completed.")
print(f"Saved to: {sinr_file}")


# ------------------------------------------------
# 16. UE MOBILITY SIMULATION
# ------------------------------------------------

# Number of time instants
time_steps = np.arange(0, 121)

# UE moves gradually from -60 degrees to +60 degrees
mobile_ue_angles = np.linspace(
    -60,
    60,
    len(time_steps)
)

mobile_selected_beams = []
mobile_best_gains = []


for ue_angle in mobile_ue_angles:

    # Channel for moving UE
    h = steering_vector(
        ue_angle,
        N
    )

    beam_gains = []

    # Test every beam
    for k in range(NUM_BEAMS):

        gain = calculate_beam_gain(
            codebook[k],
            h
        )

        beam_gains.append(gain)

    beam_gains = np.array(
        beam_gains
    )

    # Select strongest beam
    best_beam = np.argmax(
        beam_gains
    )

    # Store beam index
    mobile_selected_beams.append(
        best_beam
    )

    # Store strongest gain
    mobile_best_gains.append(
        beam_gains[best_beam]
    )


mobile_selected_beams = np.array(
    mobile_selected_beams
)

mobile_best_gains = np.array(
    mobile_best_gains
)


# ------------------------------------------------
# 17. PLOT BEAM SELECTION DURING UE MOBILITY
# ------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    time_steps,
    mobile_selected_beams,
    linewidth=2
)

plt.title(
    "Beam Selection During UE Mobility"
)

plt.xlabel("Time Step")

plt.ylabel(
    "Selected Beam Index"
)

plt.yticks(
    range(NUM_BEAMS)
)

plt.grid(True)
plt.tight_layout()

mobility_file = (
    RESULTS_DIR /
    "beam_selection_during_ue_mobility.png"
)

plt.savefig(
    mobility_file,
    dpi=300
)

plt.show()

print("\nUE mobility simulation completed.")
print(f"Saved to: {mobility_file}")


# ------------------------------------------------
# 18. BEAM SELECTION ACCURACY UNDER MOBILITY
# ------------------------------------------------

# The strongest-gain beam is treated as the reference
# or "true" beam.

true_beams = []

selected_beams_with_noise = []


# Small measurement uncertainty
measurement_noise_std = 0.02


for ue_angle in mobile_ue_angles:

    # Generate channel
    h = steering_vector(
        ue_angle,
        N
    )

    beam_gains = []

    # Calculate all beam gains
    for k in range(NUM_BEAMS):

        gain = calculate_beam_gain(
            codebook[k],
            h
        )

        beam_gains.append(gain)

    beam_gains = np.array(
        beam_gains
    )

    # Reference best beam
    true_beam = np.argmax(
        beam_gains
    )

    true_beams.append(
        true_beam
    )

    # Add small measurement uncertainty
    measured_gains = (
        beam_gains +
        np.random.normal(
            0,
            measurement_noise_std,
            NUM_BEAMS
        )
    )

    # Beam selected using measured gains
    selected_beam = np.argmax(
        measured_gains
    )

    selected_beams_with_noise.append(
        selected_beam
    )


true_beams = np.array(
    true_beams
)

selected_beams_with_noise = np.array(
    selected_beams_with_noise
)


# Compare selected beam with reference beam
correct_selections = np.sum(
    true_beams ==
    selected_beams_with_noise
)

total_selections = len(
    true_beams
)

selection_accuracy = (
    correct_selections /
    total_selections
) * 100


# ------------------------------------------------
# 19. PRINT SELECTION ACCURACY
# ------------------------------------------------

print("\n" + "=" * 65)
print("BEAM SELECTION ACCURACY ANALYSIS")
print("=" * 65)

print(
    f"Correct selections : "
    f"{correct_selections}"
)

print(
    f"Total selections   : "
    f"{total_selections}"
)

print(
    f"Selection accuracy : "
    f"{selection_accuracy:.2f}%"
)

print("=" * 65)


# ------------------------------------------------
# 20. PLOT SELECTION ACCURACY
# ------------------------------------------------

# Calculate cumulative accuracy at every time step
cumulative_correct = np.cumsum(
    true_beams ==
    selected_beams_with_noise
)

cumulative_accuracy = (
    cumulative_correct /
    np.arange(
        1,
        total_selections + 1
    )
) * 100


plt.figure(figsize=(10, 6))

plt.plot(
    time_steps,
    cumulative_accuracy,
    linewidth=2
)

plt.axhline(
    selection_accuracy,
    linestyle="--",
    linewidth=1.5,
    label=f"Final Accuracy = {selection_accuracy:.2f}%"
)

plt.title(
    "Beam Selection Accuracy Under UE Mobility"
)

plt.xlabel("Time Step")

plt.ylabel(
    "Selection Accuracy (%)"
)

plt.ylim(0, 105)

plt.grid(True)

plt.legend()

plt.tight_layout()

accuracy_file = (
    RESULTS_DIR /
    "selection_accuracy_under_ue_mobility.png"
)

plt.savefig(
    accuracy_file,
    dpi=300
)

plt.show()

print(
    f"\nSelection accuracy graph saved to: "
    f"{accuracy_file}"
)
