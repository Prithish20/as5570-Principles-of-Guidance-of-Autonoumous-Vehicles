import numpy as np
import matplotlib.pyplot as plt

r = 3000

v_T = 300.0
nu = 1.5
v_P = nu * v_T

theta = np.deg2rad(30.0)
alpha_T = np.deg2rad(170.0)
alpha_T_dot = -np.pi/6.0 

delta = 0.0
alpha_P = theta + delta

t = 0.0
dt = 0.01
Tmax = float("inf")

vx_T = v_T * np.cos(alpha_T)
vy_T = v_T * np.sin(alpha_T)
vx_P = v_P * np.cos(alpha_P)
vy_P = v_P * np.sin(alpha_P)

x_P = 0
y_P = 0
x_T = r * np.cos(theta)
y_T = r * np.sin(theta)

CR = 3  # Capture Radius

# Storage
x_T_points, y_T_points = [], []
x_P_points, y_P_points = [], []
R_points, theta_points = [], []
v_theta_points, v_R_points = [], []
a_M_points = []

min_R = float("inf")

R = np.hypot(x_T - x_P, y_T - y_P)

# Simulation loop

while R > CR and t < Tmax:

    x_T_points.append(x_T)
    y_T_points.append(y_T)
    x_P_points.append(x_P)
    y_P_points.append(y_P)
    R_points.append(R)

    min_R = min(min_R, R)

    # 1) Update target heading by constant turn rate, then its velocity
    alpha_T = alpha_T + alpha_T_dot * dt
    # Optional: wrap to [-pi, pi) to keep angles bounded
    alpha_T = (alpha_T + np.pi) % (2*np.pi) - np.pi

    vx_T = v_T * np.cos(alpha_T)
    vy_T = v_T * np.sin(alpha_T)

    x_T += vx_T * dt
    y_T += vy_T * dt

    alpha_P = np.arctan2(y_T - y_P, x_T - x_P)

    vx_P = v_P * np.cos(alpha_P)
    vy_P = v_P * np.sin(alpha_P)

    x_P += vx_P * dt
    y_P += vy_P * dt

    theta = np.arctan2(y_T - y_P, x_T - x_P)  # LOS angle

    v_R = (vx_T - vx_P) * np.cos(theta) + (vy_T - vy_P) * np.sin(theta)
    v_theta = -(vx_T - vx_P) * np.sin(theta) + (vy_T - vy_P) * np.cos(theta)
    a_M = -v_R * v_theta / max(R, 1e-9)

    theta_points.append(theta)
    v_R_points.append(v_R)
    v_theta_points.append(v_theta)
    a_M_points.append(a_M)

    t += dt
    R = np.hypot(x_T - x_P, y_T - y_P)


if R_points and R_points[-1] <= CR:
    print("Target Captured at t =", round(t, 2), "s at distance", round(R_points[-1], 2), "m")
else:
    print("Target Missed!")
    print("Closest approach =", round(min_R, 2), "m at time", round(t, 2), "s")

# Plots

time = np.arange(len(R_points)) * dt

plt.figure(figsize=(12, 4))
plt.plot(time, R_points, label="R (distance)")
plt.xlabel("Time [s]")
plt.ylabel("Distance R [m]")
plt.title("Target-Pursuer Distance Over Time")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time, np.rad2deg(theta_points), label="Theta (deg)", color="orange")
plt.xlabel("Time [s]")
plt.ylabel("Theta [deg]")
plt.title("Line-of-Sight Angle Over Time")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time, v_R_points, label="v_R")
plt.plot(time, v_theta_points, label="v_theta")
plt.xlabel("Time [s]")
plt.ylabel("Velocities [m/s]")
plt.title("Radial and Angular Velocities")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(time, a_M_points, label="a_M", color="r")
plt.xlabel("Time [s]")
plt.ylabel("a_M [m/s²]")
plt.title("Pursuer Acceleration Over Time")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(8,6))

# Target trajectory
plt.plot(x_T_points, y_T_points, label="Target", color="blue")

# Pursuer trajectory
plt.plot(x_P_points, y_P_points, label="Pursuer", color="red")

# Starting positions
plt.scatter(x_T_points[0], y_T_points[0], color="blue", marker="o", label="Target Start")
plt.scatter(x_P_points[0], y_P_points[0], color="red", marker="s", label="Pursuer Start")

# Ending positions
plt.scatter(x_T_points[-1], y_T_points[-1], color="blue", marker="x", s=100, label="Target End")
plt.scatter(x_P_points[-1], y_P_points[-1], color="red", marker="x", s=100, label="Pursuer End")

plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Target vs Pursuer Trajectories")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
