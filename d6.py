'''
Author: tu155 13293356554@163.com
Date: 2025-03-18 21:24:55
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-21 17:22:31
FilePath: \项目计划\d6.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from gurobipy import Model, GRB

def solve_d6_problem(eta, phi, mu, d_bar, d_under, rho, e):
    # 创建模型
    model = Model("optimization_problem")

    # 决策变量
    d = [model.addVar(lb=d_under[i], ub=d_bar[i], vtype=GRB.CONTINUOUS, name="d"+str(i)) for i in range(len(eta))]
    u = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="u"+str(i)) for i in range(len(phi))]
    b = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="b")

    # 设置目标函数
    objective = sum(-d[i] * eta[i] for i in range(len(eta))) + sum(-u[i] * phi[i] for i in range(len(phi))) - b * rho
    model.setObjective(objective, GRB.MAXIMIZE)

    # 添加约束条件
    for i in range(len(eta)):
        model.addConstr(d[i] - d_bar[i] <= 0, name="d_upper_bound_" + str(i))
        model.addConstr(-d[i] + d_under[i] <= 0, name="d_lower_bound_" + str(i))
        model.addConstr(d[i] - mu[i] <= u[i], name="d-mu_upper_bound" + str(i))
        model.addConstr(mu[i] - d[i] <= u[i], name="d-mu_lower_bound" + str(i))

    diff_sum = sum(e[i] * (d[i] - mu[i]) for i in range(len(eta)))
    model.addConstr(diff_sum <= b, name="sum_diff_upper_bound")
    model.addConstr(-diff_sum <= b, name="sum_diff_lower_bound")

    # 求解模型
    model.optimize()

    # 输出结果
    if model.status == GRB.OPTIMAL:
        return model.objVal, [d[i].x for i in range(len(eta))], [u[i].x for i in range(len(phi))], b.x
    else:
        return None, None, None, None

if __name__ == "__main__":
    # 假设参数如下，您需要根据实际情况进行修改
    eta = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向量
    phi = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # 向量
    mu = [2, 2, 2, 2, 2, 2, 2, 2, 2]   # 向量
    d_bar = [3, 3, 3, 3, 3, 3, 3, 3, 3]  # 向量
    d_under = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 向量
    rho = 0    # 标量
    e = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # 调用函数求解
    result = solve_d6_problem(eta, phi, mu, d_bar, d_under, rho, e)
    print('result')
    print(result)
    # 输出结果
    if result:
        optimal_value, d_values, u_values, b_value = result
        print(f"Optimal value: {optimal_value}")
        for i in range(len(eta)):
            print(f"d[{i}]: {d_values[i]}")
        for i in range(len(phi)):
            print(f"u[{i}]: {u_values[i]}")
        print(f"b: {b_value}")
    else:
        print("No optimal solution found.")