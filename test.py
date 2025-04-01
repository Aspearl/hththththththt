import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# 时间步长与预测时域长度
dt = 0.1     # 离散化时间步长
N = 20       # 预测步数

# 系统状态： [x, y, vx, vy]
nx = 4
# 控制输入： [ax, ay]
nu = 2

# 状态空间模型：双积分模型
A = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
B = np.array([[0.5*dt**2, 0],
              [0, 0.5*dt**2],
              [dt, 0],
              [0, dt]])

# 初始状态（例如从原点静止出发）
x_init = np.array([0, 0, 0, 0])
# 目标状态（例如期望到达位置 (5, 5)，速度为 0）
x_ref = np.array([5, 5, 0, 0])

# 定义优化变量
x = cp.Variable((nx, N+1))
u = cp.Variable((nu, N))

# 目标函数与约束集合
cost = 0
constraints = [x[:, 0] == x_init]

# 权重矩阵（可根据实际需要调节）
Q = np.diag([10, 10, 1, 1])  # 对位置误差赋予较高权重
R = 0.1 * np.eye(nu)         # 控制输入的平滑性

for k in range(N):
    # 累计阶段代价：状态误差与控制代价
    cost += cp.quad_form(x[:, k] - x_ref, Q) + cp.quad_form(u[:, k], R)
    # 系统动力学约束
    constraints += [x[:, k+1] == A @ x[:, k] + B @ u[:, k]]
    # 可添加控制输入约束（例如加速度上限）
    constraints += [cp.norm(u[:, k], 'inf') <= 2]

# 对终端状态也添加一定的代价
cost += cp.quad_form(x[:, N] - x_ref, Q)

# 构造并求解优化问题
problem = cp.Problem(cp.Minimize(cost), constraints)
problem.solve()

# 打印最优控制输入序列
print("最优控制输入序列:")
print(u.value)

# 绘制机器人运动轨迹
x_trajectory = x.value

plt.figure(figsize=(6, 6))
plt.plot(x_trajectory[0, :], x_trajectory[1, :], 'bo-', label='轨迹')
plt.plot(x_ref[0], x_ref[1], 'rx', markersize=12, label='目标')
plt.xlabel('x')
plt.ylabel('y')
plt.title('机器人运动轨迹')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
