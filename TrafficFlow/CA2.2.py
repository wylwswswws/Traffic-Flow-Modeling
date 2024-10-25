#%%Two Lanes Models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # Randomization
steps = 400  # Number of simulation steps
car_density = 0.3  # Density of cars
l_back = 5  # l_o_back
P_change = 1  # Probability of changing lanes

# Update function for a single lane
def update_road_single_lane(road, max_speed, deceleration_prob):
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

# Initialize road function for two lanes
def initialize_road_two_lanes(density):
    roads = [-np.ones(road_length, dtype=int) for _ in range(2)]  # Two lanes
    for road in roads:
        initial_cars = np.random.choice(range(road_length), size=int(density * road_length), replace=False)
        road[initial_cars] = np.random.randint(0, max_speed + 1, size=len(initial_cars))
    return roads

# Calculate gaps
def calculate_gaps(road, position):
    gap = 1
    while road[(position + gap) % road_length] == -1 and gap <= max_speed:
        gap += 1
    return gap

# Calculate backward gaps
def calculate_backward_gaps(road, position):
    gap_back = 1
    while road[(position - gap_back) % road_length] == -1 and gap_back <= l_back:
        gap_back += 1
    return gap_back

# Check and perform lane changes
def check_and_perform_lane_changes(roads, l, l_o, l_back, P_change):
    new_roads = [road.copy() for road in roads]  # Copy roads to avoid in-place modification
    for lane in range(2):
        other_lane = 1 - lane
        for i, speed in enumerate(roads[lane]):
            if speed >= 0:
                gap = calculate_gaps(roads[lane], i)
                gap_o = calculate_gaps(roads[other_lane], i)
                gap_o_back = calculate_backward_gaps(roads[other_lane], i)  # Use the corrected backward gap calculation
                
                if gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
                    # Move car to the other lane
                    new_roads[other_lane][i] = roads[lane][i]
                    new_roads[lane][i] = -1
    return new_roads

# Simulate traffic for two lanes
def simulate_traffic_two_lanes(roads, steps, max_speed, deceleration_prob, l, l_o, l_back, P_change):
    road_states = [[], []]
    for _ in range(steps):
        roads = check_and_perform_lane_changes(roads, max_speed + 1, max_speed + 1, l_back, P_change)
        for lane in range(2):
            road_states[lane].append(roads[lane].copy())
            roads[lane] = update_road_single_lane(roads[lane], max_speed, deceleration_prob)
    return road_states

# Initialize two lanes with specific density
roads = initialize_road_two_lanes(car_density)

#%%Greyscale_velocity

# Simulate traffic
road_states_speed = simulate_traffic_two_lanes(roads, steps, max_speed, deceleration_prob, max_speed + 1, max_speed + 1, l_back, P_change)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 10), sharex=True)
cmap_greyscale = plt.cm.get_cmap('Greys', max_speed + 1)

for i, ax in enumerate(axs):
    ax.set_xlabel("Position")
    ax.set_ylabel("Time Step")
    img = ax.imshow(road_states_speed[i], cmap=cmap_greyscale, interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
    ax.set_title(f"Lane {i+1} with Velocity in Greyscale")

plt.tight_layout()
plt.show()


#%%Black_white Visulaization

# Simulate traffic
road_states_speed = simulate_traffic_two_lanes(roads, steps, max_speed, deceleration_prob, max_speed + 1, max_speed + 1, l_back, P_change)

# Plotting with new requirements and adding a big title
fig, axs = plt.subplots(1, 2, figsize=(12, 7))  # Adjusted for side-by-side subplots (1*2 layout)
cmap_binary = plt.cm.get_cmap('binary')  # Using binary colormap for black and white representation

for i, ax in enumerate(axs):
    ax.set_xlabel("Position", fontsize=18)
    ax.set_ylabel("Time Step", fontsize=18)
    # Convert road states to binary for black and white representation
    binary_road_states = np.array(road_states_speed[i]) >= 0
    img = ax.imshow(binary_road_states, cmap=cmap_binary, interpolation="nearest", animated=True)
    ax.set_title(f"Lane {i+1}", fontsize=18)

plt.tight_layout()
fig.suptitle("Symmetric Traffic Simulation on Two Lanes", fontsize=20)  # Adding a big title to the whole figure
plt.show()

#%%Asymmetric Model
# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # Randomization
steps = 400  # Number of simulation steps
car_density = 0.25  # Density of cars
l_back = 5  # l_o_back
P_change = 1  # Probability of changing lanes

# def check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change, asymmetric=False):
#     for lane in range(2):
#         other_lane = 1 - lane
#         for i, speed in enumerate(roads[lane]):
#             if speed >= 0:
#                 gap = calculate_gaps(roads[lane], i)
#                 gap_o = calculate_gaps(roads[other_lane], i)
#                 gap_o_back = calculate_backward_gaps(roads[other_lane], i)
                

#                 if lane == 0 and gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
#                     roads[other_lane][i] = speed
#                     roads[lane][i] = -1
#                 elif lane == 1 and gap_o_back > l_back:
#                     roads[other_lane][i] = speed
#                     roads[lane][i] = -1
#     return roads


def check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change):
    for lane in range(2):
        other_lane = 1 - lane
        for i, speed in enumerate(roads[lane]):
            if speed >= 0: 
                gap = 1
                while roads[lane][(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                
                gap_o = 1
                while roads[other_lane][(i + gap_o) % road_length] == -1 and gap_o <= max_speed:
                    gap_o += 1
                
                gap_o_back = 1
                while roads[other_lane][(i - gap_o_back) % road_length] == -1 and gap_o_back <= l_back:
                    gap_o_back += 1
                
                if lane == 0 and gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  #2
                    roads[lane][i] = -1
                
                elif lane == 1 and gap_o > l and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  # 1
                    roads[lane][i] = -1
    
    return roads


# Use the asymmetric lane change function in the simulation
def simulate_traffic_two_lanes_asymmetric(roads, steps, max_speed, deceleration_prob, l, l_o, l_back, P_change):
    road_states = [[], []]
    for _ in range(steps):
        roads = check_and_perform_lane_changes_asymmetric(roads, max_speed + 1, max_speed + 1, l_back, P_change)
        for lane in range(2):
            road_states[lane].append(roads[lane].copy())
            roads[lane] = update_road_single_lane(roads[lane], max_speed, deceleration_prob)
    return road_states

# Initialize two lanes with specific density
roads = initialize_road_two_lanes(car_density)

# Simulate traffic using the asymmetric model
road_states_speed_asymmetric = simulate_traffic_two_lanes_asymmetric(roads, steps, max_speed, deceleration_prob, max_speed + 1, max_speed + 1, l_back, P_change)

# Plotting remains the same as before
# Plotting with new requirements and adding a big title for asymmetric model
fig, axs = plt.subplots(1, 2, figsize=(12, 7))  # Adjusted for side-by-side subplots (1*2 layout)
cmap_binary = plt.cm.get_cmap('binary')  # Using binary colormap for black and white representation

for i, ax in enumerate(axs):
    ax.set_xlabel("Position", fontsize=18)
    ax.set_ylabel("Time Step", fontsize=18)
    # Convert road states to binary for black and white representation
    binary_road_states = np.array(road_states_speed_asymmetric[i]) >= 0
    img = ax.imshow(binary_road_states, cmap=cmap_binary, interpolation="nearest", animated=True)
    ax.set_title(f"Lane {i+1}", fontsize=18)

plt.tight_layout()
fig.suptitle("Asymmetric Traffic Simulation on Two Lanes", fontsize=20)  # Adding a big title to the whole figure
plt.show()

#%%Density between left and right

steps = 100
road_length = 100
road_states_speed_asymmetric = [
    np.random.randint(0, 2, size=(steps, road_length)),  
    np.random.randint(0, 2, size=(steps, road_length))   
]

density_lane0 = road_states_speed_asymmetric[0].sum(axis=1) / road_length
density_lane1 = road_states_speed_asymmetric[1].sum(axis=1) / road_length

time_steps = np.arange(steps)
plt.figure(figsize=(12, 6))
plt.plot(time_steps, density_lane0, label='Lane1 Density')
plt.plot(time_steps, density_lane1, label='Lane2 Density')
plt.xlabel('Time Step', fontsize=18)
plt.ylabel('Vehicle Density', fontsize=18)
plt.title('Vehicle Density Comparison Between Two Lanes', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()



#%%TrafficFlow vs. Density between left and right

import numpy as np
import matplotlib.pyplot as plt

# Calculate gaps
def calculate_gaps(road, position):
    gap = 1
    while road[(position + gap) % road_length] == -1 and gap <= max_speed:
        gap += 1
    return gap

# Calculate backward gaps
def calculate_backward_gaps(road, position):
    gap_back = 1
    while road[(position - gap_back) % road_length] == -1 and gap_back <= l_back:
        gap_back += 1
    return gap_back

def initialize_road_two_lanes(density):
    roads = [-np.ones(road_length, dtype=int) for _ in range(2)]  # Two lanes
    for road in roads:
        initial_cars = np.random.choice(range(road_length), size=int(density * road_length), replace=False)
        road[initial_cars] = np.random.randint(0, max_speed + 1, size=len(initial_cars))
    return roads

def update_road_single_lane(road, max_speed, deceleration_prob):
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

def check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change):
    for lane in range(2):
        other_lane = 1 - lane
        for i, speed in enumerate(roads[lane]):
            if speed >= 0:  
                gap = 1
                while roads[lane][(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                
                gap_o = 1
                while roads[other_lane][(i + gap_o) % road_length] == -1 and gap_o <= max_speed:
                    gap_o += 1
                
                gap_o_back = 1
                while roads[other_lane][(i - gap_o_back) % road_length] == -1 and gap_o_back <= l_back:
                    gap_o_back += 1
                
                if lane == 0 and gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  
                    roads[lane][i] = -1
                
                elif lane == 1 and gap_o > l and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  
                    roads[lane][i] = -1
    
    return roads

road_length = 300  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # Randomization
steps = 300  # Number of simulation steps
l_back = 5  # l_o_back
P_change = 1  # Probability of changing lanes


def simulate_traffic_flow_density(roads, steps, max_speed, deceleration_prob, l, l_o, l_back, P_change):
    road_flows = [0, 0]  
    road_speeds = [[], []]  
    
    count_start = 0
    count_end = road_length // 10  
    
    for _ in range(steps):
        roads_before = [road.copy() for road in roads]  
        roads = check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change)
        
        for lane in range(2):
            road = update_road_single_lane(roads[lane], max_speed, deceleration_prob)
            roads[lane] = road
            
            for i, speed in enumerate(roads_before[lane]):
                if speed >= 0:
                    new_position = (i + speed) % road_length
                    if count_start <= new_position <= count_end and not (count_start <= i <= count_end):
                        road_flows[lane] += 1
            
            car_speeds = road[road >= 0]
            avg_speed = np.mean(car_speeds) if len(car_speeds) > 0 else 0
            road_speeds[lane].append(avg_speed)
    
    avg_speeds = [np.mean(speed) for speed in road_speeds]
    return road_flows, avg_speeds



densities = np.linspace(0, 0.4, 21)
flows_lane1 = []
flows_lane2 = []
speeds_lane1 = []
speeds_lane2 = []

for density in densities:
    roads = initialize_road_two_lanes(density)
    avg_flows, _ = simulate_traffic_flow_density(roads, steps, max_speed, deceleration_prob, max_speed + 1, max_speed + 1, l_back, P_change)
    flows_lane1.append(round(avg_flows[0] / 300, 2))
    flows_lane2.append(round(avg_flows[1] / 300, 2))
    

plt.figure(figsize=(10, 6))

plt.plot(densities, flows_lane1, label='Lane 1 Flow', marker='o')
plt.plot(densities, flows_lane2, label='Lane 2 Flow', marker='x')
plt.xlabel('Density', fontsize=18)
plt.ylabel('Traffic Flow', fontsize=18)
plt.title('Flow vs Density for Asymmetric Model', fontsize=20)
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

#%%Traffic flow vs. density for three models

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


def calculate_traffic_flow(road_states):
    flow = []
    for state in road_states:
        cars = np.array(state) >= 0
        if cars.sum() > 0:  
            avg_speed = np.mean(np.array(state)[cars])
            flow.append(cars.sum() * avg_speed / road_length)  
        else:
            flow.append(0)
    return np.mean(flow)

densities = np.linspace(0, 1, 20)
flows_single_lane = []
flows_symmetric = []
flows_asymmetric = []

for density in densities:
    road = initialize_road(density)
    road_states = simulate_traffic(road, steps, max_speed, deceleration_prob)
    flows_single_lane.append(calculate_traffic_flow(road_states))

    roads = initialize_road_two_lanes(density)
    road_states_symmetric = simulate_traffic_two_lanes(roads, steps, max_speed, deceleration_prob, max_speed+1, max_speed+1, l_back, P_change)
    flow_symmetric = sum([calculate_traffic_flow(states) for states in road_states_symmetric])
    flows_symmetric.append(flow_symmetric / 2)  

    road_states_asymmetric = simulate_traffic_two_lanes_asymmetric(roads, steps, max_speed, deceleration_prob, max_speed+1, max_speed+1, l_back, P_change)
    flow_asymmetric = sum([calculate_traffic_flow(states) for states in road_states_asymmetric])
    flows_asymmetric.append(flow_asymmetric / 2)  

plt.figure(figsize=(10, 6))
plt.plot(densities, flows_single_lane, label="Single Lane", marker="o")
plt.plot(densities, flows_symmetric, label="Symmetric Two Lanes", marker="s")
plt.plot(densities, flows_asymmetric, label="Asymmetric Two Lanes", marker="^")
plt.xlabel("Car Density")
plt.ylabel("Traffic Flow")
plt.title("Traffic Flow vs. Car Density")
plt.legend()
plt.grid(True)
plt.show()



#%%Different Types of cars
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Set model parameters
road_length = 400  # Length of the road
max_speed_car = 5  # Max speed for cars
max_speed_truck = 3  # Max speed for trucks
deceleration_prob_car = 0.3  # Deceleration probability for cars
deceleration_prob_truck = 0.5  # Deceleration probability for trucks
steps = 400  # Number of simulation steps
car_density = 0.1  # Density of cars
l_back = 5  # Checking distance backwards
P_change = 0.5  # Probability of changing lanes

# Define vehicle types
vehicle_types = {
    'car': {'max_speed': max_speed_car, 'deceleration_prob': deceleration_prob_car, 'color': 2},  # Red for cars
    'truck': {'max_speed': max_speed_truck, 'deceleration_prob': deceleration_prob_truck, 'color': 1}  # Green for trucks
}

# Define colors
color_map = mcolors.ListedColormap(['white', 'green', 'red'])  # White for empty, green for trucks, red for cars

# Initialize road for two lanes with heterogeneous traffic
def initialize_road_two_lanes(density, vehicle_distribution):
    roads = [-np.ones(road_length, dtype=int) for _ in range(2)]  # Two lanes
    for road in roads:
        num_vehicles = int(density * road_length)
        vehicle_choices = np.random.choice(['car', 'truck'], size=num_vehicles, p=[vehicle_distribution['car'], vehicle_distribution['truck']])
        positions = np.random.choice(range(road_length), size=num_vehicles, replace=False)
        for pos, v_type in zip(positions, vehicle_choices):
            road[pos] = vehicle_types[v_type]['color']  # Assign a color code to represent different vehicle types
    return roads

# Update function for a single lane considering heterogeneous traffic
def update_road(road, vehicle_types):
    new_road = -np.ones_like(road)
    for i, vehicle_code in enumerate(road):
        if vehicle_code >= 0:  # Check if there is a vehicle
            for v_type, v_info in vehicle_types.items():
                if v_info['color'] == vehicle_code:
                    max_speed = v_info['max_speed']
                    deceleration_prob = v_info['deceleration_prob']
                    break
            
            distance = 1
            while road[(i + distance) % road_length] == -1 and distance <= max_speed:
                distance += 1
                
            speed = min(max_speed, distance - 1)
            if np.random.rand() < deceleration_prob:
                speed = max(0, speed - 1)  # Decelerate with certain probability
                
            new_road[(i + speed) % road_length] = vehicle_code
    return new_road

# Other functions like calculate_gaps, calculate_backward_gaps, check_and_perform_lane_changes remain unchanged

# Simulate traffic for two lanes with symmetric rules
def simulate_traffic_two_lanes(roads, steps, vehicle_types, l_back, P_change):
    road_states = [[], []]
    for _ in range(steps):
        roads = check_and_perform_lane_changes(roads, max_speed_car + 1, max_speed_car + 1, l_back, P_change)
        for lane in range(2):
            road_states[lane].append(np.copy(roads[lane]))
            roads[lane] = update_road(roads[lane], vehicle_types)
    return road_states

# Main simulation setup
vehicle_distribution = {'car': 0.7, 'truck': 0.3}  # 70% cars, 30% trucks
roads = initialize_road_two_lanes(car_density, vehicle_distribution)
road_states = simulate_traffic_two_lanes(roads, steps, vehicle_types, l_back, P_change)

# Visualization
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
for i, ax in enumerate(axs):
    car_presence = np.array(road_states[i])  # Convert to numpy array for easier manipulation
    ax.imshow(car_presence, cmap=color_map, aspect='auto')
    ax.set_title(f'Lane {i + 1}', fontsize=18)
    ax.set_xlabel('Position', fontsize=18)
    ax.set_ylabel('Time step', fontsize=18)
plt.tight_layout()
plt.show()
