#%%Two Lanes Models
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # Randomization
steps = 400  # Number of simulation steps
car_density = 0.5  # Density of cars
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

#%%Asymmetric Model

def check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change, asymmetric=False):
    for lane in range(2):
        other_lane = 1 - lane
        for i, speed in enumerate(roads[lane]):
            if speed >= 0:
                gap = calculate_gaps(roads[lane], i)
                gap_o = calculate_gaps(roads[other_lane], i)
                gap_o_back = calculate_backward_gaps(roads[other_lane], i)
                

                if lane == 0 and gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed
                    roads[lane][i] = -1
                elif lane == 1 and gap_o_back > l_back:
                    roads[other_lane][i] = speed
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
    ax.set_xlabel("Position")
    ax.set_ylabel("Time Step")
    # Convert road states to binary for black and white representation
    binary_road_states = np.array(road_states_speed_asymmetric[i]) >= 0
    img = ax.imshow(binary_road_states, cmap=cmap_binary, interpolation="nearest", animated=True)
    ax.set_title(f"Lane {i+1}")

plt.tight_layout()
fig.suptitle("Asymmetric Traffic Simulation on Two Lanes", fontsize=16)  # Adding a big title to the whole figure
plt.show()


#%%Density between left and right

# 示例数据：使用随机生成的数据模拟车辆密度变化
# 在实际应用中，这里应替换为模拟得到的具体车辆状态数据
steps = 100
road_length = 100
# 生成随机的车辆存在状态模拟结果，用于示例
road_states_speed_asymmetric = [
    np.random.randint(0, 2, size=(steps, road_length)),  # lane0 (右车道)
    np.random.randint(0, 2, size=(steps, road_length))   # lane1 (左车道)
]

# 计算每个时间步的车辆密度
density_lane0 = road_states_speed_asymmetric[0].sum(axis=1) / road_length
density_lane1 = road_states_speed_asymmetric[1].sum(axis=1) / road_length

# 作图对比左右两车道的车辆密度
time_steps = np.arange(steps)
plt.figure(figsize=(12, 6))
plt.plot(time_steps, density_lane0, label='Right Lane (lane0) Density')
plt.plot(time_steps, density_lane1, label='Left Lane (lane1) Density')
plt.xlabel('Time Step')
plt.ylabel('Vehicle Density')
plt.title('Vehicle Density Comparison Between Left and Right Lanes')
plt.legend()
plt.grid(True)
plt.show()
