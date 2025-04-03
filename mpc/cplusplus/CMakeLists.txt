cmake_minimum_required(VERSION 3.10)
project(mpc_tracker)

set(CMAKE_CXX_STANDARD 14)

# 查找Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# 查找OSQP
find_package(osqp REQUIRED)

# 查找OSQP-Eigen
find_package(OsqpEigen REQUIRED)

# 添加头文件路径
include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${osqp_INCLUDE_DIRS}
    ${OsqpEigen_INCLUDE_DIRS}
)

# 添加可执行文件
add_executable(mpc_tracker mpc_controller.cpp mpc_controller.cpp)

# 显式链接OSQP和OSQP-Eigen库
target_link_libraries(mpc_tracker
    ${EIGEN3_LIBRARIES}
    osqp::osqp
    OsqpEigen::OsqpEigen
)
