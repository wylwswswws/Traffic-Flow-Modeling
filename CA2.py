import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 400  # Length of the road
max_speed = 5  # v_max
deceleration_prob = 0.3  # randomization
steps = 400  # steps
l = max_speed + 1  # Look ahead on your lane
l_o = l  # Look ahead on the other lane
l_o_back = 5  # Look back on the other lane
p_change = 1  # Probability of changing lanes

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


# Update function for two lanes with lane changing rules
def update_two_lanes(road1, road2, max_speed, deceleration_prob, l, l_o, l_o_back, p_change):
    # First sub-step: Check the exchange of vehicles between the two lanes
    for road, other_road in [(road1, road2), (road2, road1)]:
        for i, speed in enumerate(road):
            if speed >= 0:
                # Calculate gaps
                gap = 1
                while road[(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                    
                gap_o = 1
                while other_road[(i + gap_o) % road_length] == -1 and gap_o <= l_o:
                    gap_o += 1
                    
                gap_o_back = 0
                while other_road[(i - gap_o_back - 1) % road_length] == -1 and gap_o_back < l_o_back:
                    gap_o_back += 1

                # Check lane changing conditions
                if gap < l and gap_o > l_o and gap_o_back > l_o_back and np.random.rand() < p_change:
                    # Move vehicle to the other lane
                    other_road[i] = speed
                    road[i] = -1

    # Second sub-step: Perform independent single-lane updates on both lanes
    new_road1 = update_road(road1, max_speed, deceleration_prob)
    new_road2 = update_road(road2, max_speed, deceleration_prob)
    
    return new_road1, new_road2

# Initialize two lanes with specific density
def initialize_two_lanes(density):
    road1 = initialize_road(density)
    road2 = initialize_road(density)
    return road1, road2

# Simulate traffic for two lanes
def simulate_traffic_two_lanes(road1, road2, steps, max_speed, deceleration_prob, l, l_o, l_o_back, p_change):
    road_states1 = []
    road_states2 = []
    for _ in range(steps):
        road_states1.append(road1.copy())
        road_states2.append(road2.copy())
        road1, road2 = update_two_lanes(road1, road2, max_speed, deceleration_prob, l, l_o, l_o_back, p_change)
    return road_states1, road_states2

# Define specific density
car_density = 0.15

# Initialize two lanes with specific density
road1, road2 = initialize_two_lanes(car_density)

# Simulate traffic for two lanes
road_states1, road_states2 = simulate_traffic_two_lanes(road1, road2, steps, max_speed, deceleration_prob, l, l_o, l_o_back, p_change)

# Adjust the colorbar position for clarity and better visualization
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# Define Grey color map
cmap_greyscale = plt.cm.get_cmap('Greys', max_speed + 1)

# Plot for lane 1
img1 = axs[0].imshow(road_states1, cmap=cmap_greyscale, interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
axs[0].set_xlabel("Position")
axs[0].set_ylabel("Time Step")
axs[0].set_title("Lane 1")

# Plot for lane 2
img2 = axs[1].imshow(road_states2, cmap=cmap_greyscale, interpolation="nearest", animated=True, vmin=-1, vmax=max_speed)
axs[1].set_xlabel("Position")
axs[1].set_title("Lane 2")

# Adjusting colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
fig.colorbar(img1, cax=cbar_ax, ticks=range(max_speed + 1)).set_label('Velocity')

plt.subplots_adjust(right=0.9)  # Adjust the right edge of the subplot to make room for the colorbar
plt.show()

#%%
# Adding a super title to the plot that spans both subplots

def plot_traffic_with_title(states1, states2, title1="Lane 1", title2="Lane 2", super_title="Traffic Simulation on Two Lanes: Symmetric"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    
    # Convert states to binary for visualization: 1 for car, 0 for no car
    binary_states1 = np.array(states1) >= 0
    binary_states2 = np.array(states2) >= 0

    # Plot for lane 1
    axs[0].imshow(binary_states1, cmap='gray', interpolation="nearest", animated=True)
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Time Step")
    axs[0].set_title(title1)

    # Plot for lane 2
    axs[1].imshow(binary_states2, cmap='gray', interpolation="nearest", animated=True)
    axs[1].set_xlabel("Position")
    axs[1].set_title(title2)

    # Adding super title
    plt.suptitle(super_title, fontsize=16)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for the super title
    plt.show()

# Plotting the traffic with super title
plot_traffic_with_title(road_states1, road_states2)

#%%asymmetric

def update_two_lanes_asymmetric(road1, road2, max_speed, deceleration_prob, l, l_o, l_o_back, p_change):
    # First sub-step: Check the exchange of vehicles between the two lanes with asymmetric behavior
    for road_index, (road, other_road) in enumerate([(road1, road2), (road2, road1)]):
        for i, speed in enumerate(road):
            if speed >= 0:
                # Calculate gaps
                gap = 1
                while road[(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                    
                gap_o = 1
                while other_road[(i + gap_o) % road_length] == -1 and gap_o <= l_o:
                    gap_o += 1
                    
                gap_o_back = 0
                while other_road[(i - gap_o_back - 1) % road_length] == -1 and gap_o_back < l_o_back:
                    gap_o_back += 1
                
                # Asymmetric behavior: cars on the left lane (road_index=1) always try to move to the right lane (road_index=0) if possible
                if road_index == 1:
                    can_change = gap_o > l_o and gap_o_back > l_o_back
                else:
                    can_change = gap < l and gap_o > l_o and gap_o_back > l_o_back and np.random.rand() < p_change

                # Check lane changing conditions
                if can_change:
                    # Move vehicle to the other lane
                    other_road[i] = speed
                    road[i] = -1

    # Second sub-step: Perform independent single-lane updates on both lanes
    new_road1 = update_road(road1, max_speed, deceleration_prob)
    new_road2 = update_road(road2, max_speed, deceleration_prob)
    
    return new_road1, new_road2


def simulate_traffic_two_lanes_asymmetric(road1, road2, steps, max_speed, deceleration_prob, l, l_o, l_o_back, p_change):
    road_states1 = []
    road_states2 = []
    for _ in range(steps):
        road_states1.append(road1.copy())
        road_states2.append(road2.copy())
        road1, road2 = update_two_lanes_asymmetric(road1, road2, max_speed, deceleration_prob, l, l_o, l_o_back, p_change)
    return road_states1, road_states2

# Initialize two lanes with specific density again to start fresh
road1, road2 = initialize_two_lanes(car_density)

# Simulate traffic for two lanes with asymmetric behavior
road_states1_asymmetric, road_states2_asymmetric = simulate_traffic_two_lanes_asymmetric(road1, road2, steps, max_speed, deceleration_prob, l, l_o, l_o_back, p_change)

# Plotting the traffic with asymmetric model and adding a super title
plot_traffic_with_title(road_states1_asymmetric, road_states2_asymmetric, "Lane 1 (Right)", "Lane 2 (Left)", "Asymmetric Traffic Simulation on Two Lanes")
#%%asymmetric
car_density = 0.15

def update_two_lanes_asymmetric_corrected(road1, road2, max_speed, deceleration_prob, l, l_o, l_o_back, p_change):
    # First sub-step: Check for lane changing with corrected asymmetric behavior
    for road_index, (road, other_road) in enumerate([(road1, road2), (road2, road1)]):
        for i, speed in enumerate(road):
            if speed >= 0:
                # Calculate gaps in the current and opposite lanes
                gap = 1
                while road[(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                    
                gap_o = 1
                while other_road[(i + gap_o) % road_length] == -1 and gap_o <= l_o:
                    gap_o += 1
                    
                gap_o_back = 0
                while other_road[(i - gap_o_back - 1) % road_length] == -1 and gap_o_back < l_o_back:
                    gap_o_back += 1

                # Asymmetric lane changing logic
                # Cars on the left lane always try to move to the right lane if it's safe
                if road_index == 1 and gap_o > l_o and gap_o_back > l_o_back:  # Moving from left to right lane
                    other_road[i] = speed
                    road[i] = -1
                # Cars on the right lane move to the left only if they have to and it's safe
                elif road_index == 0 and gap < l and gap_o > l_o and gap_o_back > l_o_back and np.random.rand() < p_change:
                    other_road[i] = speed
                    road[i] = -1

    # Second sub-step: Perform independent single-lane updates on both lanes
    new_road1 = update_road(road1, max_speed, deceleration_prob)
    new_road2 = update_road(road2, max_speed, deceleration_prob)
    
    return new_road1, new_road2

# Re-initialize two lanes with specific density and simulate with corrected asymmetric behavior
road1, road2 = initialize_two_lanes(car_density)
road_states1_asymmetric_corrected, road_states2_asymmetric_corrected = simulate_traffic_two_lanes_asymmetric(road1, road2, steps, max_speed, deceleration_prob, l, l_o, l_o_back, p_change)

# Plotting the corrected asymmetric traffic simulation
plot_traffic_with_title(road_states1_asymmetric_corrected, road_states2_asymmetric_corrected, "Lane 1 (Right)", "Lane 2 (Left)", "Corrected Asymmetric Traffic Simulation on Two Lanes")

