
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
    ax.set_ylabel(f'Density={density:.2f}')
    ax.set_yticks([])

axes[-1].set_xlabel('Position on Road')
plt.suptitle('Traffic Simulation with Cellular Automata for Different Densities')
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to not overlap the title
plt.show()