# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM
```
     To simulate the signal sampling using i) Ideal Sampling  ii) Natural Sampling  iii) Flat Top Sampling
```

# SOFTWARE REQUIRED
```
    Collab software
```

# ALGORITHMS
```
Step 1: Initialize Parameters
Set sampling frequency fs (e.g., 100 Hz).

Set signal frequency f (e.g., 5 Hz).

Define time duration for the signal (e.g., 1 second).

Step 2: Generate Continuous-Time Signal
Create time vector t using np.arange(0, 1, 1/fs).

Generate the continuous signal as a sine wave:
signal = sin(2π * f * t).

Step 3: Plot Continuous Signal
Plot the continuous-time sine wave using matplotlib.pyplot.plot.

Step 4: Perform Impulse Sampling
Use the same time vector t_sampled = t (or generate again).

Evaluate the sine function at sampled points:
signal_sampled = sin(2π * f * t_sampled).

Visualize the sampled signal using stem plot to simulate impulses.

Step 5: Reconstruct Signal from Sampled Data
Use scipy.signal.resample() to interpolate sampled signal back to the length of the original signal.

reconstructed_signal = resample(signal_sampled, len(t)).

Step 6: Plot Reconstructed Signal
Overlay the reconstructed signal (dashed line) on top of the original continuous signal.

Use a legend to distinguish between original and reconstructed signals.

```
# PROGRAM
```
#Impulse Sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
fs = 500
t = np.arange(0, 1, 1/fs) 
f = 8
signal = np.sin(2 * np.pi * f * t)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal')
plt.title('Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
t_sampled = np.arange(0, 1, 1/fs)
signal_sampled = np.sin(2 * np.pi * f * t_sampled)
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')
plt.title('Sampling of Continuous Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
reconstructed_signal = resample(signal_sampled, len(t))
plt.figure(figsize=(10, 4))
plt.plot(t, signal, label='Continuous Signal', alpha=0.7)
plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')
plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.show()
```
```
#Natural sampling
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
# Parameters
fs = 2000  # Sampling frequency (samples per second)
T = 2 # Duration in seconds
t = np.arange(0, T, 1/fs)  # Time vector
# Message Signal (sine wave message)
fm = 5  # Frequency of message signal (Hz)
message_signal = np.sin(2 * np.pi * fm * t)
# Pulse Train Parameters
pulse_rate = 50  # pulses per second
pulse_train = np.zeros_like(t)
# Construct Pulse Train (rectangular pulses)
pulse_width = int(fs / pulse_rate / 2)
for i in range(0, len(t), int(fs / pulse_rate)):
pulse_train[i:i+pulse_width] = 1
# Natural Sampling
nat_signal = message_signal * pulse_train
# Reconstruction (Demodulation) Process
sampled_signal = nat_signal[pulse_train == 1]
# Create a time vector for the sampled points
sample_times = t[pulse_train == 1]
# Interpolation - Zero-Order Hold (just for visualization)
reconstructed_signal = np.zeros_like(t)
for i, time in enumerate(sample_times):
    index = np.argmin(np.abs(t - time))
    reconstructed_signal[index:index+pulse_width] = sampled_signal[i]
# Low-pass Filter (optional, smoother reconstruction)
def lowpass_filter(signal, cutoff, fs, order=5):
nyquist = 0.5 * fs
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype='low', analog=False)
return lfilter(b, a, signal)
reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)
plt.figure(figsize=(14, 10))
# Original Message Signal
plt.subplot(4, 1, 1)
plt.plot(t, message_signal, label='Original Message Signal')
plt.legend()
plt.grid(True)
# Pulse Train
plt.subplot(4, 1, 2)
plt.plot(t, pulse_train, label='Pulse Train')
plt.legend()
plt.grid(True)
# Natural Sampling
plt.subplot(4, 1, 3)
plt.plot(t, nat_signal, label='Natural Sampling')
plt.legend()
plt.grid(True)
# Reconstructed Signal
plt.subplot(4, 1, 4)
plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
```
#Flat Top Sampling
import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 10            # Sampling frequency (Hz)
f_signal = 1       # Signal frequency (Hz)
duration = 2       # Duration of the signal (seconds)
fs_high = 1000     # High-resolution sampling for original signal

# Time vectors
t = np.arange(0, duration, 1/fs_high)
t_sampled = np.arange(0, duration, 1/fs)

# Original signal (e.g., sine wave)
signal = np.sin(2 * np.pi * f_signal * t)
sampled_signal = np.sin(2 * np.pi * f_signal * t_sampled)

# Flat-top sampled signal
flat_top_signal = np.zeros_like(t)
for i in range(len(t_sampled) - 1):
    start_index = np.where(t >= t_sampled[i])[0][0]
    end_index = np.where(t >= t_sampled[i + 1])[0][0]
    flat_top_signal[start_index:end_index] = sampled_signal[i]
# Last sample
flat_top_signal[end_index:] = sampled_signal[-1]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(t, signal, label='Original Signal', linestyle='--', alpha=0.6)
plt.step(t, flat_top_signal, label='Flat-Top Sampled Signal', where='post', linewidth=2)
plt.stem(t_sampled, sampled_signal, basefmt=" ", linefmt='r-', markerfmt='ro', label='Sample Points')
plt.title('Flat-Top Sampling')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
```

# OUTPUT
Ideal sampling
![Screenshot 2025-04-09 183620](https://github.com/user-attachments/assets/b73af3c3-a6b2-4bc4-b60d-0c327a36ed49)
![Screenshot 2025-04-09 183628](https://github.com/user-attachments/assets/13778864-67b7-4a45-80a0-93e9e81cdcd5)

Natural sampling
![Screenshot 2025-04-09 183846](https://github.com/user-attachments/assets/089d1298-88f8-47c0-8006-6e4bf89b2f1d)

Flat-top sampling
![Screenshot 2025-04-09 162446](https://github.com/user-attachments/assets/822e5fba-d41a-450f-b942-99854b09b2e1)

# RESULT / CONCLUSIONS
```
Thus the simulation of various types of sampling such as i) Ideal Sampling ii) Natural Sampling iii)Flat Top Sampling were verified.
```

