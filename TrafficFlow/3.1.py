#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# from 2.2_gipps.py
# driver model


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%################################
import numpy as np
import matplotlib.pyplot as plt

#Function of simulation with Acceleration and Deceleration in Matrix

# Function to update bn based on the given condition
def b_hat_DA(b_n):
    max_array = [max(a, b) for a, b in zip(b_n, (b_n - 3) / 2)]
    return max_array    

def calculate_new_speed(v_n, v_n_lead, s_n, a_n, b_n, tau, x_n, x_n_lead, b_h, Vn):
    vp1 = v_n + 2.5*a_n*tau*(1- v_n/Vn)*(0.025+v_n/Vn)**0.5
    vp2 = b_n*tau + np.sqrt((b_n*tau)**2 - b_n*((2*(x_n_lead - s_n - x_n)% road_length) - v_n*tau-v_n_lead**2/b_h))
    return min(vp1, vp2)


# Simulation loop
def simulation_loop_DA( positions, speeds, target_speeds, time_steps, N, road_length, a_n, b_n, s_n, tau):
    for t in range(1, time_steps):
        for i in range(N):
            # The lead vehicle is the one in front of the current vehicle
            lead_vehicle_index = (i + 1) % N
            v_n = speeds[t-1, i]
            v_n_lead = speeds[t-1, lead_vehicle_index]
            x_n = positions[t-1, i]
            x_n_lead = positions[t-1, lead_vehicle_index]
            b_h = b_hat_DA(b_n)
            Vn = target_speeds[i]
            speeds[t, i] = calculate_new_speed(v_n, v_n_lead, s_n, a_n[i], b_n[i], tau, x_n, x_n_lead, b_h[i], Vn)
            positions[t, i] = (positions[t-1, i] + speeds[t, i] * tau) % road_length
    return positions, speeds


#%%%
# Function of simulation with an and bn fixed number.

# Function to update bn based on the given condition
def b_hat(b_n):
    return max(b_n, (b_n - 3) / 2)

def calculate_new_speed(v_n, v_n_lead, s_n, a_n, b_n, tau, x_n, x_n_lead, b_h, Vn):
    vp1 = v_n + 2.5*a_n*tau*(1- v_n/Vn)*(0.025+v_n/Vn)**0.5
    vp2 = b_n*tau + np.sqrt((b_n*tau)**2 - b_n*((2*(x_n_lead - s_n - x_n)% road_length) - v_n*tau-v_n_lead**2/b_h))
    return min(vp1, vp2)


# Simulation loop
def simulation_loop( positions, speeds, target_speeds, time_steps, N, road_length, a_n, b_n, s_n, tau):
    for t in range(1, time_steps):
        for i in range(N):
            # The lead vehicle is the one in front of the current vehicle
            lead_vehicle_index = (i + 1) % N
            v_n = speeds[t-1, i]
            v_n_lead = speeds[t-1, lead_vehicle_index]
            x_n = positions[t-1, i]
            x_n_lead = positions[t-1, lead_vehicle_index]
            b_h = b_hat(b_n)
            Vn = target_speeds[i]
            speeds[t, i] = calculate_new_speed(v_n, v_n_lead, s_n, a_n, b_n, tau, x_n, x_n_lead, b_h, Vn)
            positions[t, i] = (positions[t-1, i] + speeds[t, i] * tau) % road_length
    return positions, speeds

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#fontsize

fontsize_title = 20
fontsize_label = 18


# Given parameters
N = 40 # Number of vehicles
road_length = 1000.0  # Length of circular road in meters
initial_speed = 20.0  # Initial speed of all vehicles in m/s\

a_n = np.random.normal(1.7, 0.3, N)
b_n = -2.0 * a_n # Most severe braking in m/s^2
s_n = 5.0  # Effective size of vehicle in meters
tau = 2/3  # Reaction time in seconds

# Initial positions of the vehicles equally spaced on the road
initial_positions = np.linspace(0, road_length, N, endpoint=False)

# Time setup
total_time = 100  # Total time of simulation in seconds
time_steps = int(total_time / tau)

# Initialize arrays to store positions and speeds of vehicles
positions = np.zeros((time_steps, N))
speeds = np.zeros((time_steps, N))
target_speeds = np.random.normal(20, 3.2, N)

# Set initial conditions
positions[0, :] = initial_positions
speeds[0, :] = initial_speed

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plotting the displacements of vehicles against time

positions, speeds = simulation_loop_DA(positions, speeds, target_speeds, time_steps, N, road_length, a_n, b_n, s_n, tau)

plt.figure(dpi=150)

for i in range(N):
    plt.scatter(np.arange(0, total_time, tau), positions[:, i],s=(2.0)) #, label=f'Vehicle {i+1}')


plt.xlabel('Time (s)', fontsize=fontsize_label)
plt.ylabel('Displacement (m)', fontsize=fontsize_label)
plt.title('Displacements of Vehicles on a closed Circular Road', fontsize=fontsize_title)
plt.yticks(fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_label)
plt.legend()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#plot flow and density

road_length = 1000.0  # Length of circular road in meters
initial_speed = 20.0  # Initial speed of all vehicles in m/s\


s_n = 5.0  # Effective size of vehicle in meters
tau = 2/3  # Reaction time in seconds

# Initial positions of the vehicles equally spaced on the road
initial_positions = np.linspace(0, road_length, N, endpoint=False)

# Time setup
total_time = 100  # Total time of simulation in seconds
time_steps = int(total_time / tau)



# Simulate multiple scenarios with different numbers of vehicles to represent different densities
road_length_km = road_length / 1000  # Convert road length to kilometers for density calculation
Nmax = 180
vehicle_counts = np.arange(20, Nmax, 2)  # Different vehicle counts to simulate different densities
a_n = np.random.normal(1.7, 0.3, Nmax)
b_n = -2.0 * a_n # Most severe braking in m/s^2

# Store results
densities = vehicle_counts / road_length_km  # Calculate densities for each vehicle count
target_speeds = np.random.normal(20, 3.2, 180)




def density_flow_DA(road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, vehicle_counts, a_n, b_n, target_speeds):
    mean_speeds = []
    flows = []
    for N in vehicle_counts:
        # Initialize arrays for this scenario
        positions = np.zeros((time_steps, N))
        speeds = np.zeros((time_steps, N))

        # Set initial conditions for this scenario
        positions[0, :] = np.linspace(0, road_length, N, endpoint=False)
        speeds[0, :] = initial_speed
        
        # Simulation loop for this scenario
        positions, speeds = simulation_loop_DA(positions, speeds, target_speeds, time_steps, N, road_length, a_n[:N], b_n[:N], s_n, tau)
        # Calculate average speed and flow for this scenario
        average_speed_km_h = np.mean(speeds) * 3.6  # Convert average speed to km/h
        mean_speeds.append(average_speed_km_h)
        density = N / road_length_km  # Density = Number of vehicles / Road length
        flow = average_speed_km_h * density  # Flow = Density * Average Speed
        flows.append(flow)
    return densities, flows, mean_speeds


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def density_flow( vehicle_counts, road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, a_n, b_n, target_speeds):
    mean_speeds = []
    flows = []
    for N in vehicle_counts:
        # Initialize arrays for this scenario
        positions = np.zeros((time_steps, N))
        speeds = np.zeros((time_steps, N))
        target_speeds_now = target_speeds[:N]
        
        # Set initial conditions for this scenario
        positions[0, :] = np.linspace(0, road_length, N, endpoint=False)
        speeds[0, :] = initial_speed
        
        # Simulation loop for this scenario
        positions, speeds = simulation_loop(positions, speeds, target_speeds_now, time_steps, N, road_length, a_n, b_n, s_n, tau)
        # Calculate average speed and flow for this scenario
        average_speed_km_h = np.mean(speeds) * 3.6  # Convert average speed to km/h
        mean_speeds.append(average_speed_km_h)
        density = N / road_length_km  # Density = Number of vehicles / Road length
        flow = average_speed_km_h * density  # Flow = Density * Average Speed
        flows.append(flow)
    return densities, flows, mean_speeds


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_n4 = np.random.normal(1.7, 0.3, Nmax)
b_n4 = -2.0 * a_n4

a_n1 = np.random.normal(2.0, 0.1, Nmax)
b_n1 = -2.0 * a_n

a_n2 = np.random.normal(1.7, 0.1, Nmax)
b_n2 = -2.0 * a_n2

a_n3 = np.random.normal(1.5, 0.1, Nmax)
b_n3 = -2.0 * a_n3

target_speeds = np.random.normal(20, 3.2, 180)
#%%%%%%

densities1, flows1, mean_speeds1 = density_flow_DA(road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, vehicle_counts, a_n, b_n, target_speeds)
densities2, flows2, mean_speeds2 = density_flow_DA(road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, vehicle_counts, a_n2, b_n2, target_speeds)
densities3, flows3, mean_speeds3 = density_flow_DA(road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, vehicle_counts, a_n3, b_n3, target_speeds)
densities4, flows4, mean_speeds4 = density_flow_DA(road_length, initial_speed, s_n, tau, total_time, time_steps, Nmax, vehicle_counts, a_n4, b_n4, target_speeds)

plt.figure(figsize=(10, 6))
plt.plot(densities1, flows1, marker='o', linestyle='-')
plt.plot(densities2, flows2, marker='o', linestyle='-')
plt.plot(densities3, flows3, marker='o', linestyle='-')
plt.plot(densities4, flows4, marker='o', linestyle='-')

plt.xlabel('Density (vehicles per km)', fontsize=fontsize_label)
plt.ylabel('Flow (vehicles per hour)', fontsize=fontsize_label)
plt.title('Traffic Flow vs. Density',   fontsize=fontsize_title)
plt.legend(['Cautious', 'Moderate', 'Aggressive', 'Mixed'], fontsize=fontsize_label)
plt.grid(True)


#%%%%%%

# Plot density against flow
plt.figure(figsize=(10, 6))
plt.plot(densities, flows, marker='o', linestyle='-', color='blue')
plt.xlabel('Density (vehicles per km)', fontsize=fontsize_label)
plt.ylabel('Flow (vehicles per hour)', fontsize=fontsize_label)
plt.title('Traffic Flow vs. Density',   fontsize=fontsize_title)
plt.yticks(fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_label)
plt.grid(True)
plt.show()

#%% 
def f1(x):
    return -0.39*x+65

def f2(x):
    return 90*np.exp(-0.01*x)-10

def f3(x):
    return 30*np.log(1.03/x)+ 160



#



plt.figure(figsize=(10, 6))
plt.scatter(densities, mean_speeds, marker='o', linestyle='-', color='red')
plt.plot(np.arange(0, 200, 5), f1(np.arange(0, 200, 5)))
plt.plot(np.arange(0, 200, 5), f2(np.arange(0, 200, 5)))
plt.plot(np.arange(0, 200, 5), f3(np.arange(0, 200, 5))) 
#plt.plot(np.arange(0, 200, 5), f4(np.arange(0, 200, 5)))
plt.legend(['simulation data','Model 1', 'Model 2', 'Model 3'], fontsize=fontsize_label)
plt.xlabel('Density (vehicles per km)', fontsize=fontsize_label)
plt.ylabel('Average Speed (km/h)', fontsize=fontsize_label)
plt.title('Average Speed vs. Density',   fontsize=fontsize_title)
plt.yticks(fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_label)
plt.grid(True)
plt.show()

# %%
#plot the speed of each vehicle against time    
plt.figure(dpi=150)
for i in range(N):
    plt.scatter(np.arange(0, total_time, tau), speeds[:,i],s=(2.0)) #, label=f'Vehicle {i+1}')
plt.xlabel('Time (s)', fontsize=fontsize_label)
plt.ylabel('Speed (m/s)',   fontsize=fontsize_label)
plt.title('Speed of Vehicles Against Time'  , fontsize=fontsize_title)
plt.yticks(fontsize=fontsize_label)
plt.xticks(fontsize=fontsize_label)
plt.legend()
plt.show()

# %%%%%
#plot eight 

