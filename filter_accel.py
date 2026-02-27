import numpy as np
import pandas as pd
from KalmanFilter import KalmanFilter
import csv

# Beginning velocity is extremely noisy. Using a range of values where I will be calling the zero update velocity function.
use_stops = True
stop_times = list(range(0, 250, 1))

# Load the raw accelerometer CSV data into a DataFrame 
df = pd.read_csv("data/linear_acceleration_2026-02-26_14.26.08.csv", comment="#")
raw_data = df[['time', 'ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].values

# Dynamic noise and time parameters computed from the raw data. 
still_var = np.mean(np.var(raw_data[:, 1:4], axis=0), axis=0)
process_noise = still_var / 1e3
dt = np.mean(np.diff(raw_data[:, 0]))

# Slice the data to omit the calibration windows
times = raw_data[:, 0]
accel_data = raw_data[:, 1:4]

kfilter = KalmanFilter(dt, system_noise=process_noise, measurement_noise=still_var)
filtered_data = []
filtered_covariances = []

bias = np.mean(raw_data[:, 1:4], axis=0)
for idx, datum in enumerate((accel_data - bias)):
    kfilter.predict()
    if use_stops and idx in stop_times: 
        kfilter.zero_velocity_update()
    else:
        kfilter.update(datum)
    filtered_data.append(kfilter.x.flatten())
    filtered_covariances.append(kfilter.P)

filtered_data = np.hstack((times.reshape(-1, 1), np.array(filtered_data)))

with open('data/filtered_data.csv', 'w', newline='') as csvfile:
    csvfile.write("time,a_x,a_y,a_z,v_x,v_y,v_z,x,y,z\n")
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerows(filtered_data)

