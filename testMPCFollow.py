import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import sys

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
x_current = np.array([0, 0, 0, 0])

# 总模拟时间步数
total_steps = 50

# 用于存储轨迹
x_trajectory = []
x_trajectory.append(x_current)

# 定义目标点随时间变化的函数，这里简单示例目标点匀速移动
def get_target_state(t):
    return np.array([1 + 0.1 * t, 1 + 0.1 * t, 0, 0])

for step in range(total_steps):
    # 获取当前时间步的目标状态
    x_ref = get_target_state(step)

    # 定义优化变量
    x = cp.Variable((nx, N+1))
    u = cp.Variable((nu, N))

    # 目标函数与约束集合
    cost = 0
    constraints = [x[:, 0] == x_current]

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

    if problem.status == cp.OPTIMAL:
        # 应用第一个控制输入
        u_opt = u.value[:, 0]
        x_current = A @ x_current + B @ u_opt
        x_trajectory.append(x_current)
    else:
        print(f"第 {step} 步优化问题求解失败，状态: {problem.status}")


x_trajectory = np.array(x_trajectory).T

# 打印最后一个状态点的坐标
print(f"最后一个状态点的坐标: ({x_trajectory[0, -1]}, {x_trajectory[1, -1]})")

# 绘制机器人运动轨迹
plt.figure(figsize=(6, 6))
plt.plot(x_trajectory[0, :], x_trajectory[1, :], 'bo-', label='轨迹')
# 绘制目标点轨迹
target_x = [get_target_state(t)[0] for t in range(total_steps)]
target_y = [get_target_state(t)[1] for t in range(total_steps)]
plt.plot(target_x, target_y, 'rx-', markersize=8, label='目标轨迹')
plt.xlabel('x')
plt.ylabel('y')
plt.title('机器人运动轨迹')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()