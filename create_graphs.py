import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("data/filtered_data.csv")
filtered_data = df[['time','a_x','a_y','a_z','v_x','v_y','v_z','x','y','z']].values
df = pd.read_csv("data/linear_acceleration_2026-02-26_14.26.08.csv", comment="#")
accel_data = df[['ax (m/s^2)', 'ay (m/s^2)', 'az (m/s^2)']].values
times = filtered_data[:,0]

# Filtered and unfiltered acceleration graphs per axis.
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(times, accel_data[:, 0], label='Raw Accel X', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 1], label='Filtered Accel X', linestyle='-')
plt.title('Acceleration over Time')
plt.ylabel('X Acceleration')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(times, accel_data[:, 1], label='Raw Accel Y', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 2], label='Filtered Accel Y', linestyle='-')
plt.ylabel('Y Acceleration')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(times, accel_data[:, 2], label='Raw Accel Z', linestyle='-', alpha=0.6)
plt.plot(times, filtered_data[:, 3], label='Filtered Accel Z', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Z Acceleration')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('graphs/acceleration_xyz.png')
plt.close()

# Velocity graphs of each axis.
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(times, filtered_data[:, 5], label='Filtered Velocity X', color='blue', linestyle='-')
plt.title('Velocity over Time')
plt.ylabel('X Velocity')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(times, filtered_data[:, 5], label='Filtered Velocity Y', color='orange', linestyle='-')
plt.ylabel('Y Velocity')
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(times, filtered_data[:, 6], label='Filtered Velocity Z', color='green', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Z Velocity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('graphs/velocity_xyz.png')
plt.close()

# X-Y position graph
plt.figure(figsize=(8, 8))
plt.plot(filtered_data[:, 7], filtered_data[:, 8], color='purple', linestyle='-', label='Estimated Position')
plt.title('Estimated X-Y Position')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('graphs/position_xy.png')
plt.close()