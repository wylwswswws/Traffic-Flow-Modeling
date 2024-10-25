#%%Density vs. Velosity
import numpy as np
import matplotlib.pyplot as plt

# 使用你的模型参数
road_length = 400
max_speed = 5
deceleration_prob = 0.3
steps = 400
delete_sites = 6

def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites):
    new_road = -np.ones_like(road)
    if road[0] == -1:
        road[0] = np.random.randint(0, max_speed)  # 以随机速度添加新车辆
    
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
    
    new_road[-delete_sites:] = -np.ones(delete_sites)  # 删除右侧的车辆
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
        speeds.append(np.mean(road[road >= 0]))  # 只计算有车的格子的速度平均值
    
    return np.nanmean(speeds)  # 返回所有步骤的平均速度，忽略NaN值

densities = np.linspace(0, 1, 50)  # 定义一系列密度值，例如从0.1到1.0
average_speeds = []

for density in densities:
    average_speed = simulate_traffic(density)
    average_speeds.append(average_speed)

# 绘制密度与平均速度的关系图
plt.figure(figsize=(10, 6))
plt.plot(densities, average_speeds, marker='o')
plt.title('Relationship between Density and Average Speed')
plt.xlabel('Density')
plt.ylabel('Average Speed')
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt

# 使用你的模型参数和函数

densities = np.linspace(0, 1, 50)  # 定义一系列密度值
average_speeds = []

for density in densities:
    average_speed = simulate_traffic(density)
    average_speeds.append(average_speed)

# 使用多项式回归拟合数据
degree = 3  # 选择多项式的度数，可以根据数据调整
coefficients = np.polyfit(densities, average_speeds, degree)
poly = np.poly1d(coefficients)

# 绘制原始数据和拟合曲线
plt.figure(figsize=(10, 6))
plt.plot(densities, average_speeds, 'o', label='Simulation Data')
plt.plot(densities, poly(densities), '-', label=f'Polyfit Degree {degree}')

plt.title('Relationship between Density and Average Speed with Polyfit')
plt.xlabel('Density')
plt.ylabel('Average Speed')
plt.legend()
plt.grid(True)
plt.show()

# 打印拟合的多项式表达式
print(f"The fitted polynomial expression for the relationship is:\n{poly}")

#%%
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

# 定义典型的函数形式

# 线性模型
def linear_model(density, vmax, rho_max):
    return vmax * (1 - density / rho_max)

# 对数模型
def log_model(density, vmax, rho_max):
    return vmax - vmax * np.log(density / rho_max + 1)

# 指数模型
def exp_model(density, vmax, rho_max):
    return vmax * np.exp(-density / rho_max)

# Greenberg模型
def greenberg_model(density, vmax, rho_max):
    return vmax * np.log(rho_max / density)

# 使用之前模拟的数据
road_length = 400
max_speed = 5
deceleration_prob = 0.3
steps = 400
delete_sites = 6

def update_road_bottleneck(road, max_speed, deceleration_prob, delete_sites):
    new_road = -np.ones_like(road)
    if road[0] == -1:
        road[0] = np.random.randint(0, max_speed)  # 以随机速度添加新车辆
    
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
    
    new_road[-delete_sites:] = -np.ones(delete_sites)  # 删除右侧的车辆
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
        speeds.append(np.mean(road[road >= 0]))  # 只计算有车的格子的速度平均值
    
    return np.nanmean(speeds)  # 返回所有步骤的平均速度，忽略NaN值

densities = np.linspace(0, 1, 50)  # 定义一系列密度值
average_speeds = []

for density in densities:
    average_speed = simulate_traffic(density)
    average_speeds.append(average_speed)

# 使用curve_fit进行拟合
params_linear, _ = curve_fit(linear_model, densities, average_speeds, p0=[max_speed, 1])
params_log, _ = curve_fit(log_model, densities, average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))
params_exp, _ = curve_fit(exp_model, densities, average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))

# 纠正Greenberg模型拟合时的错误
# 由于 densities > 0 可能导致索引问题，我们需要正确地过滤出大于0的密度值及其对应的平均速度

# 过滤出大于0的密度值及其对应的平均速度
positive_densities = densities[densities > 0]
positive_average_speeds = np.array(average_speeds)[densities > 0]

# 重新进行Greenberg模型的拟合
params_greenberg, _ = curve_fit(greenberg_model, positive_densities, positive_average_speeds, p0=[max_speed, 1], bounds=(0, np.inf))

# 打印所有模型的拟合参数
params_linear, params_log, params_exp, params_greenberg

#%%Plot
# 绘制模拟数据以及各个模型的预测曲线

plt.figure(figsize=(12, 8))

# 模拟数据
plt.scatter(densities, average_speeds, color='black', label='Simulation Data')

# 线性模型
predicted_speeds_linear = linear_model(densities, *params_linear)
plt.plot(densities, predicted_speeds_linear, label='Linear Model')

# 对数模型
predicted_speeds_log = log_model(densities, *params_log)
plt.plot(densities, predicted_speeds_log, label='Log Model')

# 指数模型
predicted_speeds_exp = exp_model(densities, *params_exp)
plt.plot(densities, predicted_speeds_exp, label='Exp Model')

# Greenberg模型
predicted_speeds_greenberg = greenberg_model(positive_densities, *params_greenberg)
plt.plot(positive_densities, predicted_speeds_greenberg, label='Greenberg Model')

plt.title('Comparison of Traffic Models', fontsize=20)
plt.xlabel('Density', fontsize=18)
plt.ylabel('Average Speed', fontsize=18)
plt.legend()
plt.grid(True)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter

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

# Start the animation code
car_density = 0.3  # Initial density of cars
road_states = simulate_traffic(road_length, car_density, steps, max_speed, deceleration_prob, delete_sites)  # Use the simulation function

# Set up the figure and axis for the animation
fig, ax = plt.subplots(figsize=(15, 6))
ax.set_title("Traffic Simulation")
ax.set_xlabel("Position")
ax.set_ylabel("Time Step")
img = ax.imshow(road_states, cmap='gray', interpolation='nearest', animated=True)
plt.colorbar(img, ax=ax, ticks=range(max_speed + 1), label='Velocity')

def animate(i):
    ax.clear()
    ax.set_title("Traffic Simulation - Step: " + str(i))
    ax.set_xlabel("Position")
    ax.set_ylabel("Time Step")
    ax.imshow([road_states[i]], cmap='gray', interpolation='nearest', animated=True)

# Recreate animation with the corrected function
ani = animation.FuncAnimation(fig, animate, frames=len(road_states), interval=200)

# Save the corrected animation as a GIF instead for compatibility
ani.save('/Users/yulunwu/Downloads/traffic_simulation1.gif', writer=PillowWriter(fps=10))

plt.close(fig)  # Prevent the last frame from displaying inline

'/Users/yulunwu/Downloads/traffic_simulation1.gif'


#%%

