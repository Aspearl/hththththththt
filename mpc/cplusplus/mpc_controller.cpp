#include <iostream>
#include <Eigen/Dense>
#include <OsqpEigen/OsqpEigen.h>

class MPCTracker {
private:
    // MPC参数
    int horizon;              // 预测时域长度
    double dt;                // 时间步长
    int nx;                   // 状态维度 (x,y,vx,vy)
    int nu;                   // 控制输入维度 (ax,ay)
    
    // 权重矩阵
    Eigen::MatrixXd Q;        // 状态跟踪误差权重
    Eigen::MatrixXd R;        // 控制输入权重
    
    // 系统约束
    double max_acceleration;  // 最大加速度
    
    // OSQP求解器
    OsqpEigen::Solver solver;
    
    // 系统动态矩阵
    Eigen::MatrixXd A;        // 状态转移矩阵
    Eigen::MatrixXd B;        // 控制输入矩阵
    
public:
    MPCTracker(int horizon_len = 10, double time_step = 0.1) 
        : horizon(horizon_len), dt(time_step), nx(4), nu(2) {
        
        // 初始化权重矩阵
        Q = Eigen::MatrixXd::Identity(nx, nx);
        Q(0,0) = 10.0;  // x位置权重
        Q(1,1) = 10.0;  // y位置权重
        Q(2,2) = 1.0;   // x速度权重
        Q(3,3) = 1.0;   // y速度权重
        
        R = Eigen::MatrixXd::Identity(nu, nu);
        R(0,0) = 0.1;   // x加速度权重
        R(1,1) = 0.1;   // y加速度权重
        
        // 设置约束
        max_acceleration = 5.0;  // 最大加速度
        
        // 初始化系统动态矩阵
        initSystemDynamics();
        
        // 设置OSQP求解器
        initOSQPSolver();
    }
    
    void initSystemDynamics() {
        // 离散时间系统模型: x_{k+1} = A*x_k + B*u_k
        // 状态向量: [x, y, vx, vy]
        // 控制输入: [ax, ay]
        
        A = Eigen::MatrixXd::Identity(nx, nx);
        A(0,2) = dt;  // x' = x + vx*dt
        A(1,3) = dt;  // y' = y + vy*dt
        
        B = Eigen::MatrixXd::Zero(nx, nu);
        B(2,0) = dt;  // vx' = vx + ax*dt
        B(3,1) = dt;  // vy' = vy + ay*dt
    }
    
    void initOSQPSolver() {
        // 设置QP问题的维度
        solver.data()->setNumberOfVariables(nx * (horizon + 1) + nu * horizon);
        solver.data()->setNumberOfConstraints(nx * (horizon + 1));
        
        // 允许warm start
        solver.settings()->setWarmStart(true);
        
        // 设置初始化选项
        solver.settings()->setVerbosity(false);
        solver.settings()->setMaxIteration(1000);
    }

    Eigen::VectorXd solve(const Eigen::VectorXd &current_state,
                                      const Eigen::VectorXd &target_trajectory)
    {

        int n_variables = nx * (horizon + 1) + nu * horizon;
        int n_constraints = nx * (horizon + 1);

        // 首次运行时初始化矩阵
        static bool firstRun = true;

        if (!firstRun)
        {
            // 清除之前设置的矩阵和求解器
            solver.clearSolver();
            solver.data()->clearHessianMatrix();
            solver.data()->clearLinearConstraintsMatrix();
        }

        // 设置目标函数: 0.5 * x' * P * x + q' * x
        Eigen::SparseMatrix<double> hessian_matrix(n_variables, n_variables);
        Eigen::VectorXd gradient_vector = Eigen::VectorXd::Zero(n_variables);

        // 构建Hessian矩阵
        std::vector<Eigen::Triplet<double>> hessian_triplets;

        // 状态权重
        for (int i = 0; i < horizon + 1; i++)
        {
            for (int j = 0; j < nx; j++)
            {
                int idx = i * nx + j;
                hessian_triplets.push_back(Eigen::Triplet<double>(idx, idx, Q(j, j)));
            }
        }

        // 控制输入权重
        for (int i = 0; i < horizon; i++)
        {
            for (int j = 0; j < nu; j++)
            {
                int idx = (horizon + 1) * nx + i * nu + j;
                hessian_triplets.push_back(Eigen::Triplet<double>(idx, idx, R(j, j)));
            }
        }

        hessian_matrix.setFromTriplets(hessian_triplets.begin(), hessian_triplets.end());

        // 设置梯度向量 - 目标跟踪
        for (int i = 0; i < horizon + 1; i++)
        {
            int target_idx = i * nx;
            if (target_idx + nx <= target_trajectory.size())
            {
                Eigen::VectorXd target_state = target_trajectory.segment(target_idx, nx);
                for (int j = 0; j < nx; j++)
                {
                    gradient_vector(i * nx + j) = -Q(j, j) * target_state(j);
                }
            }
        }

        // 设置约束: l <= A*x <= u
        Eigen::SparseMatrix<double> constraint_matrix(n_constraints, n_variables);
        Eigen::VectorXd lower_bound = Eigen::VectorXd::Zero(n_constraints);
        Eigen::VectorXd upper_bound = Eigen::VectorXd::Zero(n_constraints);

        std::vector<Eigen::Triplet<double>> constraint_triplets;

        // 初始状态约束
        for (int j = 0; j < nx; j++)
        {
            constraint_triplets.push_back(Eigen::Triplet<double>(j, j, 1.0));
            lower_bound(j) = current_state(j);
            upper_bound(j) = current_state(j);
        }

        // 系统动态约束
        for (int i = 0; i < horizon; i++)
        {
            // x_{k+1} = A*x_k + B*u_k
            for (int j = 0; j < nx; j++)
            {
                // -I * x_{k+1}
                constraint_triplets.push_back(
                    Eigen::Triplet<double>((i + 1) * nx + j, (i + 1) * nx + j, -1.0));

                // A * x_k
                for (int k = 0; k < nx; k++)
                {
                    constraint_triplets.push_back(
                        Eigen::Triplet<double>((i + 1) * nx + j, i * nx + k, A(j, k)));
                }

                // B * u_k
                for (int k = 0; k < nu; k++)
                {
                    constraint_triplets.push_back(
                        Eigen::Triplet<double>((i + 1) * nx + j, (horizon + 1) * nx + i * nu + k, B(j, k)));
                }
            }
        }

        constraint_matrix.setFromTriplets(constraint_triplets.begin(), constraint_triplets.end());

        // 初始化求解器
        if (firstRun)
        {
            solver.data()->setNumberOfVariables(n_variables);
            solver.data()->setNumberOfConstraints(n_constraints);
            solver.settings()->setWarmStart(true);
            solver.settings()->setVerbosity(false);
            solver.settings()->setMaxIteration(1000);
            firstRun = false;
        }

        // 设置QP问题
        solver.data()->setHessianMatrix(hessian_matrix);
        solver.data()->setGradient(gradient_vector);
        solver.data()->setLinearConstraintsMatrix(constraint_matrix);
        solver.data()->setLowerBound(lower_bound);
        solver.data()->setUpperBound(upper_bound);

        // 设置控制输入约束 (如果需要)
        // 这部分保持不变...

        // 初始化求解器
        if (!solver.initSolver())
        {
            std::cerr << "无法初始化OSQP求解器" << std::endl;
            return Eigen::VectorXd::Zero(nu);
        }

        // 求解问题
        if (!solver.solve())
        {
            std::cerr << "无法求解QP问题" << std::endl;
            return Eigen::VectorXd::Zero(nu);
        }

        // 获取优化结果
        Eigen::VectorXd solution = solver.getSolution();

        // 返回第一个控制输入
        return solution.segment((horizon + 1) * nx, nu);
    }

    Eigen::VectorXd predictTargetTrajectory(const Eigen::VectorXd& target_state, int steps) {
        // 基于目标当前状态预测其未来轨迹
        // 假设目标以恒速运动
        Eigen::VectorXd trajectory = Eigen::VectorXd::Zero(nx * steps);
        
        Eigen::VectorXd current = target_state;
        for (int i = 0; i < steps; i++) {
            // 更新位置: p' = p + v*dt
            current(0) += current(2) * dt;
            current(1) += current(3) * dt;
            
            // 将当前状态添加到轨迹中
            trajectory.segment(i * nx, nx) = current;
        }
        
        return trajectory;
    }
};

// 使用示例
int main() {
    // 创建MPC追踪器
    MPCTracker tracker(20, 0.1);  // 预测时域20步，时间步长0.1秒
    
    // 初始状态 [x, y, vx, vy]
    Eigen::VectorXd current_state(4);
    current_state << 0.0, 0.0, 0.0, 0.0;
    
    // 目标状态 [x, y, vx, vy]
    Eigen::VectorXd target_state(4);
    target_state << 10.0, 5.0, 1.0, 0.5;
    
    // 模拟追踪
    double sim_time = 10.0;  // 模拟10秒
    int sim_steps = sim_time / 0.1;
    
    for (int i = 0; i < sim_steps; i++) {
        // 预测目标轨迹
        Eigen::VectorXd target_trajectory = tracker.predictTargetTrajectory(target_state, 21);
        
        // 计算最优控制输入
        Eigen::VectorXd control = tracker.solve(current_state, target_trajectory);
        
        // 更新当前状态 (简单欧拉积分)
        current_state(2) += control(0) * 0.1;  // vx' = vx + ax*dt
        current_state(3) += control(1) * 0.1;  // vy' = vy + ay*dt
        current_state(0) += current_state(2) * 0.1;  // x' = x + vx*dt
        current_state(1) += current_state(3) * 0.1;  // y' = y + vy*dt
        
        // 更新目标状态 (简单欧拉积分)
        target_state(0) += target_state(2) * 0.1;
        target_state(1) += target_state(3) * 0.1;
        
        // 输出当前状态和目标状态
        std::cout << "时间: " << i * 0.1 << "s" << std::endl;
        std::cout << "当前位置: (" << current_state(0) << ", " << current_state(1) << ")" << std::endl;
        std::cout << "目标位置: (" << target_state(0) << ", " << target_state(1) << ")" << std::endl;
        std::cout << "距离: " << 
            sqrt(pow(current_state(0) - target_state(0), 2) + 
                pow(current_state(1) - target_state(1), 2)) << std::endl;
        std::cout << "控制输入: (" << control(0) << ", " << control(1) << ")" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }
    
    return 0;
}
