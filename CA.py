#%%Setting of Single lane Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 100  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # randomization
steps = 100  # steps

# Update function
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for i, speed in enumerate(road):
        if speed >= 0:
            # Calculate distance to the next car
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1
                
            # Acceleration
            if speed < max_speed and distance > speed + 1:
                speed += 1
            
            # Slowing down
            speed = min(speed, distance - 1)

            # Randomization
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            # Car motion
            new_road[(i + speed) % road_length] = speed
    return new_road

# Initialize road function
def initialize_road(density):
    road = -np.ones(road_length, dtype=int)  # -1 represents no car
    initial_cars = np.random.choice(range(road_length), size=int(density * road_length), replace=False)
    road[initial_cars] = np.random.randint(0, max_speed + 1, size=len(initial_cars))
    return road

# Simulate traffic function
def simulate_traffic(road, steps, max_speed, deceleration_prob):
    road_states = []
    for _ in range(steps):
        road_states.append(road.copy())
        road = update_road(road, max_speed, deceleration_prob)
    return road_states

#%%Plot Single lane with closed system
# Define specific density
car_density = 0.3

# Initialize road with specific density
road = initialize_road(car_density)

# Simulate traffic
road_states_speed = simulate_traffic(road, steps, max_speed, deceleration_prob)

# Define Grey color map
cmap_greyscale = plt.cm.get_cmap('Greys', max_speed + 1)

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlabel("Position")
ax.set_ylabel("Time Step")
img = ax.imshow(road_states_speed, cmap=cmap_greyscale, interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
ax.set_title("Cellular Automation for Single Lane with Velocity in Greyscale")
colorbar = plt.colorbar(img, ticks=range(max_speed + 1), label='Velocity')
colorbar.set_label('Velocity', rotation=270, labelpad=15)

# Update plot function
def update_anim_greyscale(i):
    if i == 0:
        return img,
    img.set_array(road_states_speed[:i])
    return img,

ani_greyscale = animation.FuncAnimation(fig, update_anim_greyscale, frames=steps, interval=50, blit=True)

plt.show()

#%%Low and High Density
# Low and high density settings
low_density = 0.1
high_density = 0.7

# Initialize road for low and high density
road_low_density = initialize_road(low_density)
road_high_density = initialize_road(high_density)

# Simulate low and high density traffic
road_states_low = simulate_traffic(road_low_density, steps, max_speed, deceleration_prob)
road_states_high = simulate_traffic(road_high_density, steps, max_speed, deceleration_prob)

# Create subplots for low and high density traffic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Low density traffic
img1 = ax1.imshow(road_states_low, cmap=cmap_greyscale, interpolation="nearest", vmin=-1, vmax=max_speed)
ax1.set_title("Low Density $\\rho = 0.1$", fontsize=18)
ax1.set_xlabel("Position", fontsize=18)
ax1.set_ylabel("Time Step", fontsize=18)

# High density traffic
img2 = ax2.imshow(road_states_high, cmap=cmap_greyscale, interpolation="nearest", vmin=-1, vmax=max_speed)
ax2.set_title("High Density $\\rho = 0.7$", fontsize=18)
ax2.set_xlabel("Position", fontsize=18)
ax2.set_ylabel("Time Step", fontsize=18)

plt.tight_layout()
plt.show()


#%%Draw Car Trajectories
car_density=0.3
steps=200

def update_road_with_wrap(road, car_ids, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    new_car_ids = -np.ones_like(road)
    car_moves = {}  # Dictionary to track the moves

    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1
            if speed < max_speed and distance > speed + 1:
                speed += 1
            speed = min(speed, distance - 1)
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1
            new_position = (i + speed) % road_length
            new_road[new_position] = speed
            new_car_ids[new_position] = car_ids[i]
            car_moves[car_ids[i]] = (new_position, speed)
    return new_road, new_car_ids, car_moves


def initialize_road_with_ids(density):
    road = -np.ones(road_length, dtype=int)
    car_ids = -np.ones(road_length, dtype=int)
    initial_positions = np.random.choice(range(road_length), size=int(density * road_length), replace=False)
    initial_ids = np.arange(len(initial_positions))
    road[initial_positions] = np.random.randint(0, max_speed + 1, size=len(initial_positions))
    car_ids[initial_positions] = initial_ids
    return road, car_ids


road, car_ids = initialize_road_with_ids(car_density)

car_trajectories = {car_id: [] for car_id in car_ids if car_id != -1}

for time_step in range(1, steps + 1):
    road, car_ids, car_moves = update_road_with_wrap(road, car_ids, max_speed, deceleration_prob)
    for car_id, (new_position, speed) in car_moves.items():
        if car_trajectories[car_id] and (new_position < car_trajectories[car_id][-1][1]) and speed > 0:
            car_trajectories[car_id].append(None)  
        car_trajectories[car_id].append((time_step, new_position))

plt.figure(figsize=(12, 12))

for car_id, trajectory in car_trajectories.items():
    segments = []
    current_segment = []
    for point in trajectory:
        if point is None:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
        else:
            current_segment.append(point)
    if current_segment:  # Add the last segment
        segments.append(current_segment)
    
    for segment in segments:
        if segment:
            times, positions = zip(*segment)
            plt.plot(positions, times, lw=1)

plt.xlabel('Position on road', fontsize=18)
plt.ylabel('Time step', fontsize=18)
plt.title('Traffic Flow Trajectories with Periodic Boundary', fontsize=18)
plt.xlim(0, road_length)
plt.ylim(0, steps)
plt.gca().invert_yaxis()  # Invert the y-axis so that time increases downwards
plt.show()


#%%Traffic flow vs density
def calculate_flow(road_states, T):
    road_length = len(road_states[0])
    flow = np.zeros(road_length)

    # Calculate flow
    for t in range(T):
        for i in range(road_length):
            if road_states[t][i] > 0 and ((i + road_states[t][i]) % road_length) == (i + 1) % road_length:
                flow[i] += 1

    flow /= T

    return np.mean(flow)

densities = np.linspace(0, 1, 50)  
flow_10_steps = []
flow_1000_steps = []

for density in densities:
    # Initialize road with specific density
    road = initialize_road(density)

    # Simulate traffic for 10 and 1000 time steps
    road_states_10 = simulate_traffic(road, 100, max_speed, deceleration_prob)
    flow_10 = calculate_flow(road_states_10, 100)
    flow_10_steps.append(flow_10)

    road_states_1000 = simulate_traffic(road, 10000, max_speed, deceleration_prob)
    flow_1000 = calculate_flow(road_states_1000, 10000)
    flow_1000_steps.append(flow_1000)

# Traffic flow vs. Density
plt.figure(figsize=(10, 6))
plt.scatter(densities, flow_10_steps, label='100 time steps')
plt.plot(densities, flow_1000_steps, label='10000 time steps')
plt.xlabel('Density (cars per site)', fontsize=18)
plt.ylabel('Traffic flow (cars per time step)', fontsize=18)
plt.title('Traffic Flow vs. Density', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()

#%%Traffic flow under different densities
import numpy as np
import matplotlib.pyplot as plt

# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # Maximum speed
deceleration_prob = 0.3  # Probability of random deceleration
densities = np.linspace(0.1, 0.8, 8)  # Range of densities to simulate
steps = 400  # Number of simulation steps

# Update function for the traffic model
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1

            # Acceleration
            if speed < max_speed:
                speed += 1

            # Slowing down due to other cars
            speed = min(speed, distance - 1)

            # Random deceleration
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            # Car movement
            new_road[(i + speed) % road_length] = speed
    return new_road

# Initialize road function
def initialize_road(road_length, density):
    road = -np.ones(road_length, dtype=int)
    filled_cells = np.random.choice(road_length, size=int(density * road_length), replace=False)
    road[filled_cells] = 0  # Change here: set occupied cells to 0 (will be black)
    return road

# Simulate traffic for different densities and visualize
fig, axes = plt.subplots(len(densities), 1, figsize=(12, 2 * len(densities)), sharex=True)

for ax, density in zip(axes, densities):
    road = initialize_road(road_length, density)
    road_states = [road.copy()]
    for _ in range(steps):
        road = update_road(road, max_speed, deceleration_prob)
        road_states.append(road.copy())
    
    # Convert road states for visualization: 1 for empty, 0 for occupied
    road_states_visual = np.where(np.array(road_states) == -1, 1, 0)
    
    # Visualization
    ax.imshow(road_states_visual, cmap='gray', interpolation='nearest', aspect='auto')
    #ax.set_ylabel(f'$\\rho$={density:.2f}', fontsize=18)
    ax.text(-10, 0, f'$\\rho$={density:.2f}', fontsize=18, verticalalignment='center', horizontalalignment='right')
    ax.set_yticks([])

axes[-1].set_xlabel('Position on Road', fontsize=18)
plt.suptitle('Traffic Simulation with Cellular Automata for Different Densities', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to not overlap the title
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # Maximum speed
deceleration_prob = 0.3  # Probability of random deceleration
densities = np.linspace(0.1, 0.8, 8)  # Range of densities to simulate
steps = 400  # Number of simulation steps

# Update function for the traffic model
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1

            # Acceleration
            if speed < max_speed:
                speed += 1

            # Slowing down due to other cars
            speed = min(speed, distance - 1)

            # Random deceleration
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            # Car movement
            new_road[(i + speed) % road_length] = speed
    return new_road

# Initialize road function
def initialize_road(road_length, density):
    road = -np.ones(road_length, dtype=int)
    filled_cells = np.random.choice(road_length, size=int(density * road_length), replace=False)
    road[filled_cells] = 0
    return road

# Simulate traffic for different densities and visualize
fig, axes = plt.subplots(len(densities), 1, figsize=(12, 2 * len(densities)), sharex=True)

for ax, density in zip(axes, densities):
    road = initialize_road(road_length, density)
    road_states = [road.copy()]
    for _ in range(steps):
        road = update_road(road, max_speed, deceleration_prob)
        road_states.append(road.copy())
    
    # Convert road states for visualization: 1 for empty, 0 for occupied
    road_states_visual = np.where(np.array(road_states) == -1, 1, 0)
    
    # Visualization
    ax.imshow(road_states_visual, cmap='gray', interpolation='nearest', aspect='auto')
    # Set density label as horizontal text
    ax.text(-15, 0.5, f'$\\rho$={density:.2f}', fontsize=25, verticalalignment='center', horizontalalignment='right')

axes[-1].set_xlabel('Position on Road', fontsize=25)
plt.suptitle('Traffic Simulation with Cellular Automata for Different Densities', fontsize=25)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to not overlap the title
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

road_length = 300  # Length of the road
deceleration_prob = 0.3  # randomization
steps = 300  # Number of time steps
densities = np.linspace(0.05, 0.6, 20)  # Traffic densities to simulate
max_speeds = range(1, 6)  # Range of max speeds to test

# Update function for the road
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1

            if speed < max_speed:
                speed += 1

            speed = min(speed, distance - 1)

            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            new_road[(i + speed) % road_length] = speed
    return new_road

# Initialize road with given density
def initialize_road(density):
    road = -np.ones(road_length, dtype=int)
    car_indices = np.random.choice(road_length, size=int(road_length * density), replace=False)
    road[car_indices] = np.random.randint(0, max_speed + 1, size=len(car_indices))
    return road

# Function to simulate traffic and calculate the flow
def simulate_and_calculate_flow(max_speed, density):
    road = initialize_road(density)
    car_passes = 0
    for _ in range(steps):
        road = update_road(road, max_speed, deceleration_prob)
        car_passes += (road[0] >= 0)  # Counting cars passing the first cell
    return car_passes / steps  # Average flow per time step


# Simulate and plot the results
plt.figure(figsize=(10, 6))
for max_speed in max_speeds:
    flows = [simulate_and_calculate_flow(max_speed, density) for density in densities]
    plt.plot(densities, flows, marker='o', label=f'Max Speed {max_speed}')

plt.title('Traffic Flow vs. Density for Different Max Speeds')
plt.xlabel('Density')
plt.ylabel('Traffic Flow')
plt.legend()
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# Set model parameters
road_length = 300  # Length of the road
deceleration_prob = 0.3  # randomization
steps = 300  # simulation steps
densities = np.linspace(0, 1, 20)  # range of traffic densities

# Update function
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1
            if speed < max_speed and distance > speed + 1:
                speed += 1
            speed = min(speed, distance - 1)
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1
            new_road[(i + speed) % road_length] = speed
    return new_road

# Initialize road function
def initialize_road(density):
    road = -np.ones(road_length, dtype=int)
    initial_cars = np.random.choice(range(road_length), size=int(density * road_length), replace=False)
    road[initial_cars] = np.random.randint(0, max_speed + 1, size=len(initial_cars))
    return road

# Function to simulate traffic and calculate flow
def simulate_and_calculate_flow(density, max_speed, deceleration_prob, steps):
    road = initialize_road(density)
    road_states = []
    for _ in range(steps):
        road_states.append(road.copy())
        road = update_road(road, max_speed, deceleration_prob)
    flow = calculate_flow(road_states, steps)
    return flow

# Function to calculate flow
def calculate_flow(road_states, T):
    road_length = len(road_states[0])
    flow = np.zeros(road_length)
    for t in range(T):
        for i in range(road_length):
            if road_states[t][i] > 0 and ((i + road_states[t][i]) % road_length) == (i + 1) % road_length:
                flow[i] += 1
    return np.mean(flow) / T


# Compare traffic flow for different max_speed settings
max_speed_settings = [2, 3, 4, 5, 6]
for max_speed in max_speed_settings:
    flows = [simulate_and_calculate_flow(density, max_speed, deceleration_prob, steps) for density in densities]
    plt.plot(densities, flows, label=f'Max Speed = {max_speed}')

plt.title('Traffic Flow vs Density for Different Max Speeds')
plt.xlabel('Density')
plt.ylabel('Traffic Flow')
plt.legend()
plt.grid(True)
plt.show()


#%%
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# Linear
def linear_model(density, vmax, rho_max):
    return vmax * (1 - density / rho_max)

# Log
def log_model(density, vmax, rho_max):
    return vmax - vmax * np.log(density / rho_max + 1)

# Exp
def exp_model(density, vmax, rho_max):
    return vmax * np.exp(-density / rho_max)

# Greenberg
def greenberg_model(density, vmax, rho_max):
    return vmax * np.log(rho_max / density)


road_length = 400
max_speed = 5
deceleration_prob = 0.3
steps = 400
delete_sites = 6

def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites):
    new_road = -np.ones_like(road)
    if road[0] == -1:
        road[0] = np.random.randint(0, max_speed)  
    
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1
                
            if speed < max_speed and distance > speed + 1:
                speed += 1
            
            speed = min(speed, distance - 1)
            
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1
                
            new_position = (i + speed) % road_length
            if new_position < road_length - delete_sites:
                new_road[new_position] = speed
    
    new_road[-delete_sites:] = -np.ones(delete_sites)  
    return new_road

def simulate_traffic(density):
    road = -np.ones(road_length, dtype=int)
    for _ in range(int(density * road_length)):
        while True:
            pos = np.random.randint(road_length)
            if road[pos] == -1:
                road[pos] = np.random.randint(0, max_speed + 1)
                break
                
    speeds = []
    for _ in range(steps):
        road = update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites)
        speeds.append(np.mean(road[road >= 0]))  
    
    return np.nanmean(speeds)  

densities = np.linspace(0, 1, 50) 
average_speeds = []

for density in densities:
    average_speed = simulate_traffic(density)
    average_speeds.append(average_speed)

params_linear, _ = curve_fit(linear_model, densities, average_speeds, p0=[max_speed, 1])
params_log, _ = curve_fit(log_model, densities, average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))
params_exp, _ = curve_fit(exp_model, densities, average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))

positive_densities = densities[densities > 0]
positive_average_speeds = np.array(average_speeds)[densities > 0]

params_greenberg, _ = curve_fit(greenberg_model, positive_densities, positive_average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))

params_linear, params_log, params_exp, params_greenberg

#%%Plot

plt.figure(figsize=(12, 8))

plt.scatter(densities, average_speeds, color='black', label='Simulation Data')

predicted_speeds_linear = linear_model(densities, *params_linear)
plt.plot(densities, predicted_speeds_linear, label='Linear Model')

predicted_speeds_log = log_model(densities, *params_log)
plt.plot(densities, predicted_speeds_log, label='Log Model')

predicted_speeds_exp = exp_model(densities, *params_exp)
plt.plot(densities, predicted_speeds_exp, label='Exp Model')

predicted_speeds_greenberg = greenberg_model(positive_densities, *params_greenberg)
plt.plot(positive_densities, predicted_speeds_greenberg, label='Greenberg Model')

plt.title('Comparison of Traffic Models', fontsize=20)
plt.xlabel('Density', fontsize=18)
plt.ylabel('Average Speed', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()