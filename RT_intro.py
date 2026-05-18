import sionna.rt
import matplotlib.pyplot as plt
import mitsuba as mi
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, RadioMapSolver, subcarrier_frequencies

scene = load_scene(sionna.rt.scene.munich)