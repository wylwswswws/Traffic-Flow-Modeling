#%%Bottleneck
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Model parameters
road_length = 100  # Length of the road
max_speed = 5  # Maximum speed
deceleration_prob = 0.3  # Probability of random deceleration
steps = 100  # Number of simulation steps
delete_sites = 6  # Number of sites to delete cars at the right side

# Initialize road with specific density
def initialize_road_with_density(road_length, density=None):
    road = -np.ones(road_length, dtype=int)  # -1 represents no car
    if density is not None:
        num_cars = int(density * road_length)
        positions = np.random.choice(road_length, size=num_cars, replace=False)
        road[positions] = np.random.randint(0, max_speed + 1, size=num_cars)
    return road

# Update function with open boundary conditions
def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites):
    new_road = -np.ones_like(road)
    if road[0] == -1:
        road[0] = 0  # Occupying with a car of velocity 0 if the leftmost site is empty
    
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

# Simulate traffic
def simulate_traffic(road_length, density, steps, max_speed, deceleration_prob, delete_sites):
    road = initialize_road_with_density(road_length, density)
    road_states = []
    for _ in range(steps):
        road_states.append(road.copy())
        road = update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites)
    return road_states

# Visualization in 1*2 layout
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Simulation without initial density
road_states_bottleneck = simulate_traffic(road_length, None, steps, max_speed, deceleration_prob, delete_sites)
img1 = axs[0].imshow(road_states_bottleneck, cmap='Greys', interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
axs[0].set_title("Traffic in a Bottleneck Situation with Empty Initial Road", fontsize=16)
axs[0].set_xlabel("Position", fontsize=18)
axs[0].set_ylabel("Time Step", fontsize=18)

# Simulation with specific initial density
car_density = 0.3  # Initial density of cars
road_states_bottleneck_density = simulate_traffic(road_length, car_density, steps, max_speed, deceleration_prob, delete_sites)
img2 = axs[1].imshow(road_states_bottleneck_density, cmap='Greys', interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
axs[1].set_title("Traffic in a Bottleneck Situation with Given Initial Density", fontsize=16)
axs[1].set_xlabel("Position", fontsize=18)
axs[1].set_ylabel("Time Step", fontsize=18)

# Adjust colorbar to be shared by subplots
plt.colorbar(img2, ax=axs[1], ticks=range(max_speed + 1), label='Velocity')

plt.tight_layout()
plt.show()

#%%Traffic flow vs. Density

# Set model parameters
road_length = 500
max_speed = 5
deceleration_prob = 0.3
steps = 500
delete_sites = 6

# Modified update_road_bottleneck function to return the number of moves
def update_road_bottleneck_and_count_moves(road, max_speed, deceleration_prob, delete_sites):
    new_road = -np.ones_like(road)
    moves = 0  # Count of total moves in this step
    if road[0] == -1:
        road[0] = 0
    
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
                moves += speed  # Add speed to moves as it represents the distance moved
    
    new_road[-delete_sites:] = -np.ones(delete_sites)
    return new_road, moves

# Modified simulate_traffic function to calculate flow based on total moves
def simulate_traffic_and_calculate_flow_based_on_moves(road, steps, max_speed, deceleration_prob, delete_sites):
    total_moves = 0  # Total moves for all cars
    for _ in range(steps):
        road, moves = update_road_bottleneck_and_count_moves(road, max_speed, deceleration_prob, delete_sites)
        total_moves += moves
    flow = total_moves / (road_length * steps)  # Average flow based on total moves
    return flow

# Calculate flow for different densities
densities = np.linspace(0.05, 0.85, 41)
flows = []

for density in densities:
    road = initialize_road_with_density(road_length, density)
    flow = simulate_traffic_and_calculate_flow_based_on_moves(road, steps, max_speed, deceleration_prob, delete_sites)
    flows.append(flow)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(densities, flows, marker='o')
plt.title("Traffic Flow vs. Initial Density", fontsize=18)
plt.xlabel("Initial Density", fontsize=18)
plt.ylabel("Traffic Flow", fontsize=18)
plt.grid(True)
plt.show()



#%%Traffic Light with yellow light
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
road_length = 400  # Length of the road
max_speed = 5  # Maximum speed
deceleration_prob = 0.3  # Probability of random deceleration
steps = 400  # Number of simulation steps
delete_sites = 6  # Sites to delete cars at the right side
traffic_light_cycle = 40  # Total cycle length (green + yellow + red)
green_light_duration = 20  # Green light duration
yellow_light_duration = 5  # Yellow light duration
traffic_light_position = road_length // 2  # Position of the traffic light

# Initialize road with specific density, only on the left side of the traffic light
def initialize_road_with_density(road_length, density=None):
    road = -np.ones(road_length, dtype=int)  # -1 represents no car
    if density is not None:
        # Calculate the number of cars based on the density and the length of the road before the traffic light
        num_cars = int(density * traffic_light_position)  # Only populate left side of the road
        positions = np.random.choice(range(traffic_light_position), size=num_cars, replace=False)  # Choose positions only before the traffic light
        road[positions] = 0  # Initialize cars with velocity 0
    return road


# Determine traffic light status
def get_traffic_light_status(step):
    cycle_position = step % traffic_light_cycle
    if cycle_position < green_light_duration:
        return 'green'
    elif cycle_position < green_light_duration + yellow_light_duration:
        return 'yellow'
    else:
        return 'red'

# Update function with open boundary conditions and a traffic light
def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites, step):
    new_road = -np.ones_like(road)
    traffic_light_status = get_traffic_light_status(step)
    
    if road[0] == -1:
        road[0] = 0  # Occupying with a car of velocity 0 if the leftmost site is empty
    
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1

            if i + distance > traffic_light_position and i < traffic_light_position:
                if traffic_light_status == 'red':
                    distance = min(distance, traffic_light_position - i)
                elif traffic_light_status == 'yellow':
                    # Try to stop if possible, else pass the yellow light if too close
                    if distance - 1 < max_speed and speed > 1:
                        speed = 0

            if speed < max_speed and distance > speed + 1:
                speed += 1
            
            speed = min(speed, distance - 1)
            
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            new_position = (i + speed) % road_length
            if new_position < road_length - delete_sites:
                new_road[new_position] = speed
    
    new_road[-delete_sites:] = -np.ones(delete_sites)  # Remove cars at the end
    return new_road

# Simulate traffic with a traffic light
def simulate_traffic(road_length, density, steps, max_speed, deceleration_prob, delete_sites):
    road = initialize_road_with_density(road_length, density)
    road_states = []
    for step in range(steps):
        road_states.append(road.copy())
        road = update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites, step)
    return road_states

density = 0.3  # Example density
road_states_bottleneck = simulate_traffic(road_length, density, steps, max_speed, deceleration_prob, delete_sites)

# Convert road states for visualization: 1 for empty, 0 for occupied
road_states_visual = np.where(np.array(road_states_bottleneck) == -1, 1, 0)

# Create figure and plot
plt.figure(figsize=(12, 12))
ax = plt.gca()  # Get current axes
im = ax.imshow(road_states_visual, cmap='gray', interpolation="nearest", aspect='auto')

# Plot traffic light status
for step in range(steps):
    traffic_light_status = get_traffic_light_status(step)
    if traffic_light_status == 'green':
        ax.axhline(y=step, color='green', xmin=0.495, xmax=0.505, linewidth=2)
    elif traffic_light_status == 'yellow':
        ax.axhline(y=step, color='yellow', xmin=0.495, xmax=0.505, linewidth=2)
    elif traffic_light_status == 'red':
        ax.axhline(y=step, color='red', xmin=0.495, xmax=0.505, linewidth=2)

# Setting the title, labels and colorbar
plt.title(f"Traffic Flow with Traffic Light (Density = {density})", fontsize=20)
plt.xlabel("Position on Road", fontsize=18)
plt.ylabel("Time Step", fontsize=18)

plt.tight_layout()
plt.show()

#%%Traffic Light(Only Green and Red)
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
road_length = 400  # Length of the road
max_speed = 5  # Maximum speed
deceleration_prob = 0.3  # Probability of random deceleration
steps = 400  # Number of simulation steps
delete_sites = 6  # Sites to delete cars at the right side
traffic_light_cycle = 40  # Total cycle length (green + yellow + red)
green_light_duration = 20  # Green light duration
traffic_light_position = road_length // 2  # Position of the traffic light

# Initialize road with specific density, only on the left side of the traffic light
def initialize_road_with_density(road_length, density=None):
    road = -np.ones(road_length, dtype=int)  # -1 represents no car
    if density is not None:
        # Calculate the number of cars based on the density and the length of the road before the traffic light
        num_cars = int(density * traffic_light_position)  # Only populate left side of the road
        positions = np.random.choice(range(traffic_light_position), size=num_cars, replace=False)  # Choose positions only before the traffic light
        road[positions] = 0  # Initialize cars with velocity 0
    return road


# Determine traffic light status
def get_traffic_light_status(step):
    cycle_position = step % traffic_light_cycle
    if cycle_position < green_light_duration:
        return 'green'
    else:
        return 'red'

# Update function with open boundary conditions and a traffic light
def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites, step):
    new_road = -np.ones_like(road)
    traffic_light_status = get_traffic_light_status(step)
    
    if road[0] == -1:
        road[0] = 0  # Occupying with a car of velocity 0 if the leftmost site is empty
    
    for i, speed in enumerate(road):
        if speed >= 0:
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1

            if i + distance > traffic_light_position and i < traffic_light_position:
                if traffic_light_status == 'red':
                    distance = min(distance, traffic_light_position - i)


            if speed < max_speed and distance > speed + 1:
                speed += 1
            
            speed = min(speed, distance - 1)
            
            if speed > 0 and np.random.rand() < deceleration_prob:
                speed -= 1

            new_position = (i + speed) % road_length
            if new_position < road_length - delete_sites:
                new_road[new_position] = speed
    
    new_road[-delete_sites:] = -np.ones(delete_sites)  # Remove cars at the end
    return new_road

# Simulate traffic with a traffic light
def simulate_traffic(road_length, density, steps, max_speed, deceleration_prob, delete_sites):
    road = initialize_road_with_density(road_length, density)
    road_states = []
    for step in range(steps):
        road_states.append(road.copy())
        road = update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites, step)
    return road_states

density = 0.3  # Example density
road_states_bottleneck = simulate_traffic(road_length, density, steps, max_speed, deceleration_prob, delete_sites)

# Convert road states for visualization: 1 for empty, 0 for occupied
road_states_visual = np.where(np.array(road_states_bottleneck) == -1, 1, 0)

# Create figure and plot
plt.figure(figsize=(12, 12))
ax = plt.gca()  # Get current axes
im = ax.imshow(road_states_visual, cmap='gray', interpolation="nearest", aspect='auto')

# Plot traffic light status
for step in range(steps):
    traffic_light_status = get_traffic_light_status(step)
    if traffic_light_status == 'green':
        ax.axhline(y=step, color='green', xmin=0.495, xmax=0.505, linewidth=2)
    elif traffic_light_status == 'red':
        ax.axhline(y=step, color='red', xmin=0.495, xmax=0.505, linewidth=2)

# Setting the title, labels and colorbar
plt.title(f"Traffic Flow with Green and Red Traffic Light (Density = {density})", fontsize=20)
plt.xlabel("Position on Road", fontsize=18)
plt.ylabel("Time Step", fontsize=18)

plt.tight_layout()
plt.show()

