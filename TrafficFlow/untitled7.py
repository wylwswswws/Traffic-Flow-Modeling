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
            if speed >= 0:  # 如果当前位置有车
                # 计算当前车道前方的空位数
                gap = 1
                while roads[lane][(i + gap) % road_length] == -1 and gap <= max_speed:
                    gap += 1
                
                # 计算另一车道前方的空位数
                gap_o = 1
                while roads[other_lane][(i + gap_o) % road_length] == -1 and gap_o <= max_speed:
                    gap_o += 1
                
                # 计算另一车道后方的空位数
                gap_o_back = 1
                while roads[other_lane][(i - gap_o_back) % road_length] == -1 and gap_o_back <= l_back:
                    gap_o_back += 1
                
                # 如果车辆在车道1，并且满足换道条件
                if lane == 0 and gap < l and gap_o > l_o and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  # 变道到车道2
                    roads[lane][i] = -1
                
                # 如果车辆在车道2，并且车道1前方和后方的空位都多于车道2，满足换道回车道1的条件
                elif lane == 1 and gap_o > l and gap_o_back > l_back and np.random.rand() < P_change:
                    roads[other_lane][i] = speed  # 变道回车道1
                    roads[lane][i] = -1
    
    return roads


# 模型参数
road_length = 300  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # Randomization
steps = 300  # Number of simulation steps
l_back = 5  # l_o_back
P_change = 1  # Probability of changing lanes


def simulate_traffic_flow_density(roads, steps, max_speed, deceleration_prob, l, l_o, l_back, P_change):
    road_flows = [0, 0]  # 初始化两条车道的流量计数为0
    road_speeds = [[], []]  # 记录两条车道的平均速度
    
    # 定义计数区间的起始点和终点
    count_start = 0
    count_end = road_length // 10  # 假设计数区间为道路长度的1/10
    
    for _ in range(steps):
        roads_before = [road.copy() for road in roads]  # 更新前的车道状态
        roads = check_and_perform_lane_changes_asymmetric(roads, l, l_o, l_back, P_change)
        
        for lane in range(2):
            road = update_road_single_lane(roads[lane], max_speed, deceleration_prob)
            roads[lane] = road
            
            # 计算流量：检查哪些车辆通过了计数区间
            for i, speed in enumerate(roads_before[lane]):
                if speed >= 0:
                    new_position = (i + speed) % road_length
                    # 如果车辆的新位置超过了计数区间的起始点，并且原始位置在计数区间的起始点之前
                    if count_start <= new_position <= count_end and not (count_start <= i <= count_end):
                        road_flows[lane] += 1
            
            # 计算平均速度
            car_speeds = road[road >= 0]
            avg_speed = np.mean(car_speeds) if len(car_speeds) > 0 else 0
            road_speeds[lane].append(avg_speed)
    
    # 计算总平均速度
    avg_speeds = [np.mean(speed) for speed in road_speeds]
    return road_flows, avg_speeds



# 在一系列不同的密度值上运行模拟
densities = np.linspace(0, 0.4, 21)
flows_lane1 = []
flows_lane2 = []
speeds_lane1 = []
speeds_lane2 = []

for density in densities:
    roads = initialize_road_two_lanes(density)
    avg_flows, _ = simulate_traffic_flow_density(roads, steps, max_speed, deceleration_prob, max_speed + 1, max_speed + 1, l_back, P_change)
    # 除以300并保留两位小数
    flows_lane1.append(round(avg_flows[0] / 300, 2))
    flows_lane2.append(round(avg_flows[1] / 300, 2))
    

# 作图比较两条车道上的流量与密度关系
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


#%%
# 假设已经有了flows_lane1, flows_lane2, speeds_lane1, speeds_lane2的数据
# 计算两条车道的流量和速度的平均值
average_flows = [(flow1 + flow2) / 2 for flow1, flow2 in zip(flows_lane1, flows_lane2)]
average_speeds = [(speed1 + speed2) / 2 for speed1, speed2 in zip(speeds_lane1, speeds_lane2)]

# 使用Matplotlib进行作图
plt.figure(figsize=(10, 6))


plt.plot(densities, average_flows, label='Average Flow', marker='o', color='blue')
plt.xlabel('Density')
plt.ylabel('Average Traffic Flow')
plt.title('Average Flow vs Density')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()


print(densities)
print(average_flows)
#%%
import matplotlib.pyplot as plt

densities_0 = [0,         0.02105263, 0.04210526, 0.06315789, 0.08421053, 0.10526316,
 0.12631579, 0.14736842, 0.16842105, 0.18947368, 0.21052632, 0.23157895,
 0.25263158, 0.27368421, 0.29473684, 0.31578947, 0.33684211, 0.35789474,
 0.37894737, 0.4       ]

flow_3 = [0.0, 0.09, 0.19, 0.27, 0.37, 0.46, 0.42, 0.44, 0.43, 0.43, 0.41,0.43, 0.41, 0.4,0.39, 0.4, 0.39, 0.36, 0.36, 0.34]
flow_2 = [0.0, 0.08, 0.18, 0.25, 0.37, 0.44, 0.40, 0.44, 0.42, 0.41, 0.4 ,0.42, 0.41, 0.38,0.39, 0.39, 0.38, 0.35, 0.35, 0.33]
flow_1 = [0.0, 0.08, 0.17, 0.23, 0.34, 0.37, 0.34, 0.34, 0.32, 0.31, 0.30 ,0.31, 0.30, 0.30 ,0.29, 0.29, 0.28, 0.27, 0.25, 0.25]

plt.figure(figsize=(10, 6))


# plt.plot(densities_0, flow_3, label='Asymmetric Model', marker='o', color='blue')
# plt.plot(densities_0, flow_2, label='Symmetric Model', marker='x', color='red')
# plt.plot(densities_0, flow_1, label='Single-Lane', marker='+', color='green')
plt.plot(densities_0, flow_3, label='Asymmetric Model', marker='o')
plt.plot(densities_0, flow_2, label='Symmetric Model', marker='x')
plt.plot(densities_0, flow_1, label='Single-Lane', marker='+')
plt.xlabel('Density', fontsize=18)
plt.ylabel('Traffic Flow', fontsize=18)
plt.title('Traffic Flow vs Density for Three Models', fontsize=20)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#%%比较两个模型
import numpy as np
import matplotlib.pyplot as plt

densities2 = np.array([20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.,
                       75., 80., 85., 90., 95., 100., 105., 110., 115., 120., 125.,
                       130., 135., 140., 145., 150., 155., 160., 165., 170., 175.])
flow2 = np.array([1209.017001101906, 1573.087536143414,
                  1711.9016619884537, 2028.8620476178994, 2204.453523345458,
                  2300.250887222371, 2396.7206762756477, 2405.70045753528,
                  2659.592710640631, 2585.780610430743, 2534.450575696483,
                  2453.987204079096, np.nan, 2336.4140752340345,
                  2263.2453357598747, 2214.7894736842104, np.nan, np.nan,
                  2013.2727272727273, np.nan, np.nan, np.nan, 1740.4615384615383,
                  1671.7407407407409, np.nan, 1533.8275862068956, np.nan, np.nan,
                  np.nan, np.nan, 1186.9411595496895, np.nan])

# Filter out 'nan' values and corresponding densities
valid_indices = ~np.isnan(flow2)
filtered_densities = densities2[valid_indices]
filtered_flow = flow2[valid_indices]

densities2_1 = np.array([ 20.,  22.,  24.,  26.,  28.,  30.,  32.,  34.,  36.,  38.,  40.,
        42.,  44.,  46.,  48.,  50.,  52.,  54.,  56.,  58.,  60.,  62.,
        64.,  66.,  68.,  70.,  72.,  74.,  76.,  78.,  80.,  82.,  84.,
        86.,  88.,  90.,  92.,  94.,  96.,  98., 100., 102., 104., 106.,
       108., 110., 112., 114., 116., 118., 120., 122., 124., 126., 128.,
       130., 132., 134., 136., 138., 140., 142., 144., 146., 148., 150.,
       152., 154., 156., 158., 160., 162., 164., 166., 168., 170., 172.,
       174., 176., 178.])
                         
flow2_1 = np.array([1197.2220164778653,
 1314.096077344183,
 1431.492542972212,
 1547.1161569959097,
 1660.8272684892847,
 1770.6474961759645,
 1878.2596578679054,
 1983.7394549033809,
 2085.4651800190727,
 2183.0759542061382,
 2276.3985383313134,
 2365.113159978704,
 2447.3048046289437,
 2524.0260738848824,
 2596.2268451248747,
 2664.0378577704114,
 2725.0186254256764,
 2723.998076530564,
 2703.7866135795834,
 2681.1266608222554,
 2657.871968682817,
 2633.809203733715,
 2608.498135123951,
 2580.436499928797,
 2555.3917075514114,
 2533.5617879245583,
 2512.443403682255,
 2489.0918918859115,
 2463.536842102917,
 2437.835897435006,
 2411.999999999672,
 2386.039024390127,
 2359.9619047618644,
 2333.7767441860333,
 2307.490909090905,
 2281.1111111111095,
 2254.6434782608694,
 2228.0936170212767,
 2201.466666666667,
 2174.7673469387755,
 2148.0,
 2121.1686274509807,
 2094.276923076923,
 2067.3283018867924,
 2040.3259259259264,
 2013.2727272727273,
 1986.1714285714293,
 1959.0245614035086,
 1931.8344827586209,
 1904.6033898305086,
 1877.3333333333337,
 1850.0262295081968,
 1822.6838709677422,
 1795.3079365079366,
 1767.8999999999999,
 1740.4615384615383,
 1712.9939393939394,
 1685.4985074626866,
 1657.9764705882353,
 1630.4289855072464,
 1602.8571428571424,
 1575.2619718309845,
 1547.6444444444446,
 1520.0054794520179,
 1492.3459459459004,
 1464.6666666666615,
 1436.9684210525172,
 1409.2519480517817,
 1381.5179486761426,
 1353.7670885750265,
 1325.9999999954948,
 1298.2172839168836,
 1270.4195120040242,
 1242.60722740408,
 1214.7809519280788,
 1186.9411595496895,
 1159.0881691169802,
 1131.2226681484308,
 1103.3341778585125,
 1075.4548538756605])

# Filter out 'nan' values and corresponding densities
valid_indices = ~np.isnan(flow2_1)
filtered_densities_1 = densities2_1[valid_indices]
filtered_flow_1 = flow2_1[valid_indices]
    
flow2 = np.array([1209.017001101906, 1573.087536143414,
                  1711.9016619884537, 2028.8620476178994, 2204.453523345458,
                  2300.250887222371, 2396.7206762756477, 2405.70045753528,
                  2659.592710640631, 2585.780610430743, 2534.450575696483,
                  2453.987204079096, np.nan, 2336.4140752340345,
                  2263.2453357598747, 2214.7894736842104, np.nan, np.nan,
                  2013.2727272727273, np.nan, np.nan, np.nan, 1740.4615384615383,
                  1671.7407407407409, np.nan, 1533.8275862068956, np.nan, np.nan,
                  np.nan, np.nan, 1186.9411595496895, np.nan])

# Filter out 'nan' values and corresponding densities
valid_indices = ~np.isnan(flow2)
filtered_densities = densities2[valid_indices]
filtered_flow = flow2[valid_indices]


densities1 = [  0.      ,   5.882352,  11.764706,  17.647058,  23.529412,
        29.411764,  35.294118,  41.17647 ,  47.058824,  52.941176,
        58.82353 ,  64.705882,  70.588236,  76.470588,  82.352942,
        88.235294,  94.117648, 100.      , 105.882352, 111.764706,
       117.647058, 123.529412, 129.411764, 135.294118, 141.17647 ,
       147.058824, 152.941176, 158.82353 , 164.705882, 170.588236,
       176.470588, 182.352942, 188.235294, 194.117648, 200.      ]

original1 = np.array([0.00000e+00, 1.75000e-02, 4.50000e-02, 2.20000e-01, 2.77750e+00,
       7.18750e+00, 1.17975e+01, 1.63425e+01, 2.07475e+01, 2.59225e+01,
       3.09925e+01, 3.52275e+01, 3.78625e+01, 4.35750e+01, 4.62600e+01,
       5.31525e+01, 5.45425e+01, 5.82025e+01, 5.97050e+01, 6.07525e+01,
       6.47075e+01, 6.43925e+01, 6.03350e+01, 6.03475e+01, 6.27850e+01,
       5.55775e+01, 5.26225e+01, 4.80975e+01, 4.48975e+01, 3.72975e+01,
       3.10025e+01, 2.35525e+01, 1.65900e+01, 8.50000e+00, 1.50000e-01])

flow1 = original1  * 40

densities3 = [ 10.      ,  15.517242,  21.034482,  26.551724,  32.068966,
        37.586206,  43.103448,  48.62069 ,  54.137932,  59.655172,
        65.172414,  70.689656,  76.206896,  81.724138,  87.24138 ,
        92.75862 ,  98.275862, 103.793104, 109.310344, 114.827586,
       120.344828, 125.862068, 131.37931 , 136.896552, 142.413794,
       147.931034, 153.448276, 158.965518, 164.482758, 170.      ]

original3 = np.array([76.2832, 78.2592, 81.284 , 83.7464, 84.5456, 85.2072, 84.9984,
       84.6152, 83.232 , 80.7784, 76.0712, 77.7616, 76.4416, 73.2408,
       74.3192, 69.3448, 68.9248, 65.6488, 63.396 , 60.9184, 60.208 ,
       59.8744, 55.2816, 52.0712, 48.484 , 48.0048, 46.8736, 45.816 ,
       41.4784, 39.7536])

flow3 = original3 * 30


# Plot
plt.figure(figsize=(10, 6))
plt.plot(densities1, flow1, label='CA Model1', marker='o')
plt.plot(filtered_densities, filtered_flow, label='Car Following', marker='x')
plt.plot(densities3, flow3, label='CA Model2', marker='+')
plt.plot(densities2_1, flow2_1, label='Car Following2', marker='x')
plt.xlabel('Density')
plt.ylabel('Traffic Flow')
plt.title('Traffic Flow vs Density for Two Models')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%
# 定义原始数组
original_array_2 = np.array([0.381416, 0.391296, 0.40642, 0.418732, 0.422728, 0.426036, 0.424992, 0.423076, 0.41616, 0.403892, 0.380356, 0.388808, 0.382208, 0.366204, 0.371596, 0.346724, 0.344624, 0.328244, 0.31698, 0.304592, 0.30104, 0.299372, 0.276408, 0.260356, 0.24242, 0.240024, 0.234368, 0.22908, 0.207392, 0.198768])

# 计算每个数乘以200的结果
multiplied_array_2 = original_array_2  * 200
multiplied_array_2
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Model parameters
road_length = 100  # Length of the road
max_speed = 5  # vmax
deceleration_prob = 0.3  # randomization
steps = 100  # steps
lanes = 2  # Number of lanes

# Initialize road function for two lanes
def initialize_road_two_lanes(road_length, lanes):
    road = -np.ones((lanes, road_length), dtype=int)  # -1 represents no car
    return road

# Update function for two lanes
def update_road_two_lanes(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for lane in range(lanes):
        for i, speed in enumerate(road[lane]):
            if speed >= 0:
                # Calculate distance to the next car in the same lane
                distance = 1
                while road[lane][(i + distance) % road_length] == -1 and distance <= max_speed:
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
                new_road[lane][(i + speed) % road_length] = speed
    return new_road

# Simulate traffic function for two lanes
def simulate_traffic_two_lanes(road, steps, max_speed, deceleration_prob):
    road_states = []
    for _ in range(steps):
        road_states.append(road.copy())
        road = update_road_two_lanes(road, max_speed, deceleration_prob)
    return road_states

# Function to create an animation
def animate_traffic(road_states):
    fig, ax = plt.subplots()
    mat = ax.matshow(road_states[0], cmap='binary')
    plt.axis('off')

    def update(frame):
        mat.set_data(road_states[frame])
        return [mat]

    ani = animation.FuncAnimation(fig, update, frames=len(road_states), interval=100, blit=True)
    return ani

# Create the road and simulate traffic
road = initialize_road_two_lanes(road_length, lanes)
road_states = simulate_traffic_two_lanes(road, steps, max_speed, deceleration_prob)

# Create and display the animation
ani = animate_traffic(road_states)
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Set model parameters
road_length = 100  # Length of the road
lanes = 2  # Number of lanes
max_speed = 5  # Maximum speed
density = 0.3  # Vehicle density
steps = 50  # Number of time steps

def initialize_road(road_length, lanes, density):
    road = np.full((lanes, road_length), -1)  # Initialize empty road
    for lane in range(lanes):
        for cell in range(road_length):
            if np.random.rand() < density:
                road[lane, cell] = np.random.randint(0, max_speed + 1)  # Random initial speed
    return road

def update_road(road):
    new_road = np.full(road.shape, -1)  # Create a new empty road
    for lane in range(lanes):
        for cell in range(road_length):
            if road[lane, cell] != -1:  # If there is a car
                speed = road[lane, cell]
                # Acceleration
                speed = min(speed + 1, max_speed)
                # Slowing down (if needed)
                for ahead in range(1, speed + 1):
                    if road[lane, (cell + ahead) % road_length] != -1:
                        speed = ahead - 1
                        break
                # Randomization
                if speed > 0 and np.random.rand() < 0.3:
                    speed -= 1
                # Car movement
                new_road[lane, (cell + speed) % road_length] = speed
    return new_road

# Initialize road
road = initialize_road(road_length, lanes, density)

# Create a list to hold the state of the road at each step
roads = [road]
for _ in range(steps - 1):
    road = update_road(road)
    roads.append(road)

# Create an animation
frames = []  # for storing the generated images
for road in roads:
    fig, ax = plt.subplots()
    ax.imshow(road, cmap='binary', interpolation='nearest', aspect='auto')
    ax.set_title("Traffic simulation")
    ax.axis('off')
    # Convert figure to image
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)

# Save the frames as an animated GIF
imageio.mimsave('/Users/yulunwu/Downloads/traffic_simulation.gif', frames, fps=5)
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio

# Set model parameters
road_length = 100  # Length of each lane
lanes = 2  # Number of lanes
max_speed = 5  # Maximum speed
density = 0.3  # Vehicle density
steps = 50  # Number of time steps

# Initialize the road
road = -np.ones((lanes, road_length), dtype=int)

# Function to update road state based on CA rules
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for lane in range(lanes):
        for i in range(road_length):
            if road[lane, i] >= 0:  # There is a car in this cell
                distance = 1
                while road[lane, (i + distance) % road_length] == -1 and distance <= max_speed:
                    distance += 1
                speed = min(road[lane, i] + 1, max_speed, distance - 1)  # Acceleration, slowing down, max speed
                if np.random.rand() < deceleration_prob:  # Randomization
                    speed = max(speed - 1, 0)
                new_road[lane, (i + speed) % road_length] = speed
    return new_road

# Generate initial road state
np.random.seed(42)  # Seed for reproducibility
for lane in range(lanes):
    for i in range(road_length):
        if np.random.random() < density:
            road[lane, i] = np.random.randint(0, max_speed + 1)

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust figsize to change the width and height of the plot

# Function to animate each frame
def animate(frame_num):
    global road
    ax.clear()
    road = update_road(road, max_speed, deceleration_prob)
    ax.imshow(road, cmap='gray', aspect='auto')
    ax.set_title(f"Time step: {frame_num}")
    ax.set_ylabel("Lane")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Lane 1', 'Lane 2'])
    ax.set_xticks([])

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=steps, interval=200)

# Save animation
ani.save('/Users/yulunwu/Downloads/traffic_animation2.gif', writer='imagemagick', fps=5)

plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 100  # Length of each lane
lanes = 2  # Number of lanes
max_speed = 5  # Maximum speed
density = 0.3  # Vehicle density
steps = 50  # Number of time steps

# Initialize the road
road = -np.ones((lanes, road_length), dtype=int)

# Function to update road state based on CA rules
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for lane in range(lanes):
        for i in range(road_length):
            if road[lane, i] >= 0:  # There is a car in this cell
                distance = 1
                while road[lane, (i + distance) % road_length] == -1 and distance <= max_speed:
                    distance += 1
                speed = min(road[lane, i] + 1, max_speed, distance - 1)  # Acceleration, slowing down, max speed
                if np.random.rand() < deceleration_prob:  # Randomization
                    speed = max(speed - 1, 0)
                new_road[lane, (i + speed) % road_length] = speed
    return new_road

# Generate initial road state
np.random.seed(42)  # Seed for reproducibility
for lane in range(lanes):
    for i in range(road_length):
        if np.random.random() < density:
            road[lane, i] = np.random.randint(0, max_speed + 1)

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(10, 1))  # Reduce the height of the plot

# Function to animate each frame
def animate(frame_num):
    global road
    ax.clear()
    road = update_road(road, max_speed, deceleration_prob)
    ax.imshow(road, cmap='gray', aspect='auto')  # Change 'auto' to a higher value to make lanes narrower
    ax.set_title(f"Time step: {frame_num}")
    ax.set_ylabel("Lane")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Lane 1', 'Lane 2'])
    ax.set_xticks([])

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=steps, interval=200)

# Save animation
ani.save('/Users/yulunwu/Downloads/traffic_animation3.gif', writer='imagemagick', fps=5)

plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Set model parameters
road_length = 100  # Length of each lane
lanes = 2  # Number of lanes
max_speed = 5  # Maximum speed
density = 0.3  # Vehicle density
steps = 100  # Number of time steps

# Initialize the road
road = -np.ones((lanes, road_length), dtype=int)

# Function to update road state based on CA rules
def update_road(road, max_speed, deceleration_prob):
    new_road = -np.ones_like(road)
    for lane in range(lanes):
        for i in range(road_length):
            if road[lane, i] >= 0:  # There is a car in this cell
                distance = 1
                while road[lane, (i + distance) % road_length] == -1 and distance <= max_speed:
                    distance += 1
                speed = min(road[lane, i] + 1, max_speed, distance - 1)  # Acceleration, slowing down, max speed
                if np.random.rand() < deceleration_prob:  # Randomization
                    speed = max(speed - 1, 0)
                new_road[lane, (i + speed) % road_length] = speed
    return new_road

# Generate initial road state
np.random.seed(42)  # Seed for reproducibility
for lane in range(lanes):
    for i in range(road_length):
        if np.random.random() < density:
            road[lane, i] = np.random.randint(0, max_speed + 1)

# Create a figure for plotting
fig, ax = plt.subplots(figsize=(10, 3))  # Adjust height here to allow title to show

# Function to animate each frame
def animate(frame_num):
    global road
    ax.clear()
    road = update_road(road, max_speed, deceleration_prob)
    ax.imshow(road, cmap='gray', aspect='equal')  # Adjust aspect to change lane width
    ax.set_title(f"Time step: {frame_num}")
    ax.set_ylabel("Lane")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Lane 1', 'Lane 2'])
    ax.set_xticks([])

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=steps, interval=200)

# Save animation
ani.save('/Users/yulunwu/Downloads/traffic_animation.gif', writer='imagemagick', fps=5)

plt.tight_layout()  # Adjust layout
plt.show()

