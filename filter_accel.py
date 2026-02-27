import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from KalmanFilter import KalmanFilter
import csv

# Tunable stop_times array. Value in array represents the row in the accel_data to which we plan to perform zero update velocity.
use_stops = True
stop_times = list(range(0,280,1))

# Obtain samples of acceleration data where I stood still. I stood still in the beginning and end of the recording.
still_sample = 0

# Load the raw accelerometer CSV data into a DataFrame 
df = pd.read_csv("data/linear_acceleration_2026-02-26_14.26.08.csv", comment="#")
raw_data = df[['time', 'ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].values

start_stationary_data = raw_data[:still_sample, 1:4]
end_stationary_data = raw_data[-still_sample:, 1:4]
full_stationary_data = np.concatenate((start_stationary_data, end_stationary_data), axis=0)

# Finding mean of stationary samples to obtain bias estimate (removing factors like gravity).
bias = np.mean(full_stationary_data, axis=0)

# Dynamic measurement and process noise parameters. 
still_var = np.mean(np.var(full_stationary_data, axis=0), axis=0) * 2
process_noise = still_var / 1e4

# Calculating dt by taking the mean of differences in the time column of raw CSV data.
dt = np.mean(np.diff(raw_data[:, 0]))

# Slice the data to omit the calibration windows
times = raw_data[still_sample:, 0]
accel_data = raw_data[still_sample:, 1:4]

kfilter = KalmanFilter(dt, system_noise=process_noise, measurement_noise=still_var)
filtered_data = []
filtered_covariances = []

for idx, datum in enumerate((accel_data - bias)):
    kfilter.predict()
    # use zero velocity update on the 2 seconds we were still. 
    if use_stops and idx in stop_times: 
        kfilter.zero_velocity_update()
    else:
        kfilter.update(datum)
    filtered_data.append(kfilter.x.flatten())
    filtered_covariances.append(kfilter.P)

filtered_data = np.array(filtered_data)


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
plt.figure(figsize=(8, 8))
plt.plot(filtered_data[:, 6], filtered_data[:, 7], color='purple', linestyle='-', label='Estimated Position')
plt.title('Estimated X-Y Position Trajectory')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('xy_position_separate.png')
plt.close()

# --- 3. Velocity Graph (Separately, No Points) ---
plt.figure(figsize=(10, 8))

# Velocity X
plt.subplot(3, 1, 1)
plt.plot(times, filtered_data[:, 3], label='Filtered Velocity X', color='blue', linestyle='-')
plt.title('Velocity over Time')
plt.ylabel('X Velocity')
plt.legend()
plt.grid(True)

# Velocity Y
plt.subplot(3, 1, 2)
plt.plot(times, filtered_data[:, 4], label='Filtered Velocity Y', color='orange', linestyle='-')
plt.ylabel('Y Velocity')
plt.legend()
plt.grid(True)

# Velocity Z
plt.subplot(3, 1, 3)
plt.plot(times, filtered_data[:, 5], label='Filtered Velocity Z', color='green', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Z Velocity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('velocity_separate.png')
plt.close()


