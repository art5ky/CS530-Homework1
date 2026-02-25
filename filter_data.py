import numpy
import pandas as pd
import io
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter


# --- Data Loading and Processing ---

df = pd.read_csv("data/Accelerometer.csv")

# Format data into a matrix: [seconds_elapsed, x, y, z] 
# (Note: CSV has z, y, x, so we reorder to x, y, z for standard convention)
raw_data = df[['seconds_elapsed', 'x', 'y', 'z']].values

# 2. Setup Variables
# Because the dataset is small (11 rows), we will only use 2 rows for the still time.
still_time = 100 
process_noise = 0.001
still_var = 1000

# Calculate mean dt by looking at the differences in the time column
mean_dt = numpy.mean(numpy.diff(raw_data[:, 0]))

# Find gravity and other biases during a time period where you don't move
bias = numpy.mean(numpy.concatenate((raw_data[:still_time, 1:4], raw_data[-still_time:, 1:4])), axis=0)

# Slice the data to omit the calibration windows
times = raw_data[still_time:-still_time, 0]
accel_data = raw_data[still_time:-still_time, 1:4]

# 3. Apply Kalman Filter
kfilter = KalmanFilter(mean_dt, system_noise=process_noise, measurement_noise=still_var)
filtered_data = []
filtered_covariances = []

# Subtract the bias, which hopefully includes gravity
for datum in (accel_data - bias):
    kfilter.predict()
    kfilter.update(datum)
    # kfilter.x is shape (9,1). flatten() turns it into a 1D array of 9 elements.
    filtered_data.append(kfilter.x.flatten())
    filtered_covariances.append(kfilter.P)

filtered_data = numpy.array(filtered_data)

# --- 1. Acceleration Graph (Separately, No Points) ---
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(times, accel_data[:, 0] - bias[0], label='Raw Accel X', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 0], label='Filtered Accel X', linestyle='-')
plt.title('Acceleration over Time (No Points)')
plt.ylabel('X Accel')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(times, accel_data[:, 1] - bias[1], label='Raw Accel Y', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 1], label='Filtered Accel Y', linestyle='-')
plt.ylabel('Y Accel')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(times, accel_data[:, 2] - bias[2], label='Raw Accel Z', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 2], label='Filtered Accel Z', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Z Accel')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('acceleration_separate.png')
plt.close()

# --- 2. X-Y Position Graph (Separately) ---
plt.figure(figsize=(8, 6))
plt.plot(filtered_data[:, 6], filtered_data[:, 7], color='purple', linestyle='-', label='Estimated Position')
plt.title('Estimated X-Y Position Trajectory')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xy_position_separate.png')
plt.close()

print("Successfully generated separate plots.")