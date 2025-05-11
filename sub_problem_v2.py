from gurobipy import Model, GRB
from typing import Dict, List, Union, Tuple
import numpy as np
import itertools
from append_data import cacu_n_laf

class OptimizationError(Exception):
    """优化求解过程中的自定义错误"""
    pass

def solve_optimization_problem(
    A: list,
    I: range,
    M: range,
    T: Dict[int, List[int]],
    U: Dict[int, List[int]],
    gamma: float,
    D_a: dict,
    cacu_need:dict,
    mu: List[float],
    sigma: List[float],
    omega: float,
    d_bar: List[float],
    d_under: List[float],
    e: List[int]
) -> Dict[str, Union[float, List[float]]]:
    """
    求解优化问题
    
    参数:
        A: 项目索引范围
        I: 活动索引范围
        M: 第一阶段生成路径索引范围
        T: T_l 集合，路径l中活动索引
        U: U_l 集合，除去路径中活动之外的活动索引
        gamma: 置信度参数
        D: 最大完工时间
        mu: 活动均值列表
        sigma: 活动标准差列表
        omega: 活动波动总和
        d_bar: 活动持续时间上限列表
        d_under: 活动持续时间下限列表
        e: 参数e列表（取值均为1）
        
    返回:
        包含最优解的字典，如果无解则抛出异常
    """
    
    activities=cacu_need['activities']#活动
    activity_f=cacu_need['activity_f']#活动类型
    try:
        # 创建模型
        model = Model("S2_Model")
        
        # 定义变量
        eta = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="eta")
        phi = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="phi")
        rho = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="rho")
        theta = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="theta")
        beta = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="beta")
        for num in range(0,len(A)+1):#从0到|A|的数量
            if num==0:
                # 定义 l∈M 对应的变量组
                p_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="p")
                q_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="q")
                c_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="c")
                v_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="v")
                m_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="m")
                f_0 = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="f")
                # 添加约束条件
                # 约束 (21)
                for l in M:
                    d_bar_p = sum(d_bar[i] * p_0[l, i] for i in I)
                    d_under_q = sum(d_under[i] * q_0[l, i] for i in I)
                    mu_c = sum(mu[i] * c_0[l, i] for i in I)
                    mu_v = sum(mu[i] * v_0[l, i] for i in I)
                    e_mu_m = sum(e[i] * mu[i] * m_0[l, i] for i in I)
                    e_mu_f = sum(e[i] * mu[i] * f_0[l, i] for i in I)
                    rhs = d_bar_p - d_under_q + mu_c - mu_v + e_mu_m - e_mu_f
                    model.addConstr(theta + gamma * beta >= rhs)

                # 约束 (22)
                for l in M:
                    for j in T[l]:
                        
                        model.addConstr(
                            p_0[l, j] - q_0[l, j] + c_0[l, j] - v_0[l, j] + m_0[l, j] - f_0[l, j] >=- eta[j]         
                        )
            else:
                # 选择所有指定数量元素的组合
                element_combinations = list(itertools.combinations(A, num))

                print("所有组合：")
                print(element_combinations)
                p={}
                q={}
                c={}
                v={}
                m={}
                f={}
                #有多少组合就生成多少个p,q,c,v,m,f中的元素
                for idx,comb in enumerate(element_combinations):
                    # 定义 l∈M 对应的变量组
                    p[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="p")
                    q[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="q")
                    c[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="c")
                    v[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="v")
                    m[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="m")
                    f[idx] = model.addVars(M, I, lb=0, vtype=GRB.CONTINUOUS, name="f")
                for idx,comb in enumerate(element_combinations):
                    # 添加约束条件
                    # 约束 (21)
                    for l in M:
                        d_bar_p = sum(d_bar[i] * p[idx][l, i] for i in I)
                        d_under_q = sum(d_under[i] * q[idx][l, i] for i in I)
                        mu_c = sum(mu[i] * c[idx][l, i] for i in I)
                        mu_v = sum(mu[i] * v[idx][l, i] for i in I)
                        e_mu_m = sum(e[i] * mu[i] * m[idx][l, i] for i in I)
                        e_mu_f = sum(e[i] * mu[i] * f[idx][l, i] for i in I)
                        rhs = d_bar_p - d_under_q + mu_c - mu_v + e_mu_m - e_mu_f
                        model.addConstr(theta + sum(D_a[a] for a in comb) + gamma * beta >= rhs)

                    # 约束 (22)
                    for l in M:
                        for j in T[l]:
                            activity=activities[j]
                            if activity in activity_f:
                                type_f=activity_f[activity]
                                if type_f=='MAKE':
                                    n_lf=num*1/6
                                else:
                                    n_lf=num*2/5
                            else:
                                n_lf=0
                            model.addConstr(
                                p[idx][l, j] - q[idx][l, j] + c[idx][l, j] - v[idx][l, j] + m[idx][l, j] - f[idx][l, j] >= num +n_lf- eta[j]
                            )
                    # 约束 (23)
                    for l in M:
                        for g in U[l]:
                            model.addConstr(
                                p[idx][l, g] - q[idx][l, g] + c[idx][l, g] - v[idx][l, g] + m[idx][l, g] - f[idx][l, g] >= -eta[g]
                            )

                    # 约束 (24)
                    for l in M:
                        for i in I:
                            model.addConstr(c[idx][l, i] + v[idx][l, i] <= phi[i])

                    # 约束 (25)
                    for l in M:
                        for i in I:
                            model.addConstr(m[idx][l, i] + f[idx][l, i] <= rho)

        # 定义其他变量
        h = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="h")
        w = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="w")
        t = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="t")
        o = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="o")
        zeta = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="zeta")
        epsilon = model.addVars(I, lb=0, vtype=GRB.CONTINUOUS, name="epsilon")

        # 设置目标函数
        mu_T_eta = sum(mu[i] * eta[i] for i in I)
        sigma_T_phi = sum(sigma[i] * phi[i] for i in I)
        omega_rho = omega * rho
        model.setObjective((mu_T_eta + sigma_T_phi + omega_rho + theta) / (1 - gamma), GRB.MINIMIZE)

        

        
        # 约束 (26) 和 (27)
        d_bar_h = sum(d_bar[i] * h[i] for i in I)
        d_under_w = sum(d_under[i] * w[i] for i in I)
        mu_t = sum(mu[i] * t[i] for i in I)
        mu_o = sum(mu[i] * o[i] for i in I)
        e_mu_zeta = sum(e[i] * mu[i] * zeta[i] for i in I)
        e_mu_epsilon = sum(e[i] * mu[i] * epsilon[i] for i in I)
        rhs_common = d_bar_h - d_under_w + mu_t - mu_o + e_mu_zeta - e_mu_epsilon
        model.addConstr(theta + gamma * beta >= rhs_common)
        model.addConstr(theta - (1 - gamma) * beta >= rhs_common)

        # 约束 (28)
        for i in I:
            model.addConstr(
                h[i] - w[i] + t[i] - o[i] + zeta[i] - epsilon[i] >= -eta[i]
            )

        # 约束 (29) 和 (30)
        for i in I:
            model.addConstr(t[i] + o[i] <= phi[i])
            model.addConstr(zeta[i] + epsilon[i] <= rho)

        # 求解模型
        model.optimize()

        # 检查求解状态并返回结果
        if model.Status == GRB.OPTIMAL:
            return {
                "objective_value": model.ObjVal,
                "eta": [eta[i].x for i in I],
                "phi": [phi[i].x for i in I],
                "rho": rho.x,
                "theta": theta.x,
                "beta": beta.x
            }
        else:
            raise OptimizationError("未找到最优解")

    except Exception as e:
        raise OptimizationError(f"优化求解过程中发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 示例数据
    I = range(6)
    M = range(3)
    T = {0:[0,2,3,5], 1:[0,1,4,5], 2:[0,1,2,3]}
    U = {0:[1,4], 1:[2,3], 2:[4,5]}
    gamma = 0.9
    D = 50
    mu = [1.0] * len(I)
    sigma = [1.0] * len(I)
    omega = 10.0
    d_bar = [1.0] * len(I)
    d_under = [0.5] * len(I)
    e = [1] * len(I)

    try:
        result = solve_optimization_problem(
            I, M, T, U, gamma, D, mu, sigma, omega, d_bar, d_under, e
        )
        print("最优解：")
        for key, value in result.items():
            print(f"{key}: {value}")
    except OptimizationError as e:
        print(f"错误: {str(e)}")
    