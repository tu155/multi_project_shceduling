'''
Author: tu155 13293356554@163.com
Date: 2025-03-12 17:22:31
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-05-07 09:54:46
FilePath: \项目计划\main_prob.py
Description: gurobi求解主问题
'''
import sys

# 增加最大递归深度
sys.setrecursionlimit(10000)  # 根据实际情况调整数值
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import copy
import time
from datetime import date,timedelta,datetime
import pandas as pd
import json
import pro_pre_data
from get_exl_data import resource as re_demand,durance as du,relationship_dict as relationship,bom,point,project_acti
from get_exl_data import resource_capacity as capacity,have_worked as worked,due_time,project_status as status,type_dict as activity_f
from append_data import D_a

from sub_problem_v2 import solve_optimization_problem,OptimizationError
from get_critical_path import find_critical_paths

from cycle_test import is_cyclic

from d6 import solve_d6_problem

#计算最底层活动集合，合同是否生效
def data_process():
    '''
    计算最底层活动，返回最顶层活动项目-活动、紧前关系
    '''
    #时间范围
    du_range={}
    for i,dur_time in du.items():
        du_range[i]=[dur_time-1,dur_time,dur_time+3]

    print('data is ready')
    #计算最底层活动
    # from pro_pre_data import relationship,bom,status,point
    bottom,father,start_end,top,belong,laywer=pro_pre_data.pre_data(relationship,bom)
    bottom_relationship=pro_pre_data.transfer(relationship,bom)
    
    print('bottom_relationship')
    print(bottom_relationship)

    #使用集合的交集来获得projects
    projects=list(project_acti.keys())
    projects = list(set(project_acti.keys()) & set(status.keys()))
    print('projects')
    print(projects)
    random.seed(20)

    priority={}
    for i in projects:
        priority[i]=random.random()

    #计算每个项目合同生效之后的活动有哪些    
    next_acti=pro_pre_data.get_next_acti(bottom_relationship)
    res={}
    for key in projects:
        if key in point:
            all_successors = pro_pre_data.get_all_successors(point[key], next_acti)
            res[key]=all_successors
        else:
            res[key]=[]
    print('activities in effect of each project')
    print(res)
    
    #移除合同未生效订单的生产等活动
    belong_copy=copy.deepcopy(belong)
    for p in projects:
        if status[p]==False:
            if p in res:
                for a in belong_copy[p]:
                    if a in list(res[p]):
                        belong[p].remove(a)
    print('共读取了{}个项目'.format(len(projects)))
    return belong,bottom_relationship

#加入虚拟活动，生成新的紧前关系表
def get_plus_relationship(bottom_relationship):
    plus_relationship={}
    plus_relationship['end']=[]
    for i in bottom_relationship:
        if bottom_relationship[i]==[]:
            plus_relationship[i]=['start']
        else:
            plus_relationship[i]=bottom_relationship[i]
        if all(i not in value for value in bottom_relationship.values()):
            plus_relationship['end'].append(i)
    return plus_relationship

#计算活动持续时间期望值、标准差、期望值、期望值下限
def ambigulity(activities):
    import random
    random.seed(20)
    #时间范围
    du_range={}
    dur_time_list=[]
    for i in activities:
        if i=='start' or i=='end':
            dur_time=0
            dur_time_list.append(dur_time)
            du_range[i]=[0,dur_time,0]
        else:
            dur_time=du[i]
            dur_time=int(dur_time*random.uniform(1,1.5))
            dur_time_list.append(dur_time)
            bound=int(0.8*random.uniform(1,1.25)*dur_time) if int(0.8*random.uniform(1,1.25)*dur_time)>0 else 1
            du_range[i]=[bound,dur_time,int(dur_time+1.2*random.uniform(1,3))]
    mean=[]
    std_dev=[]
    d_bar=[]
    d_under=[]
    for activity in du_range:
        # 正态分布的期望值和标准差
        mean.append((du_range[activity][0] + 4 * du_range[activity][1] +
                du_range[activity][2]) / 6)
        std_dev.append((du_range[activity][2] - du_range[activity][0]) / 6)
        d_bar.append(du_range[activity][2])
        d_under.append(du_range[activity][0])
    return mean,std_dev,d_bar,d_under,dur_time_list
def data_to_excel(activities,mean,std_dev,d_bar,d_under,dur_time_list):
    '''
    将mean,std_dev,d_bar,d_under写入excel文件
    '''
    df=pd.DataFrame({'activities':activities,'d_under':d_under,'d_middle':dur_time_list,'d_bar':d_bar,'mean':mean,'std_dev':std_dev,})
    df.to_excel(f'input/data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx',index=False)
def get_omiga(std_dev):
    '''
    计算协方差矩阵
    '''
    co_matrix=np.zeros((len(std_dev),len(std_dev)))
    for i in range(len(std_dev)):
        for j in range(len(std_dev)):
            if i!=j:
                co_matrix[i,j]=1*std_dev[i]*std_dev[j]
            else:
                co_matrix[i,j]=std_dev[i]**2
            
    omiga=np.sqrt(np.sum(co_matrix))
    return omiga

#调整问题规模
def adjust_problem_scale(belong,bottom_relationship,project_index_set):
    '''
    belong: 活动隶属关系{项目名称:活动列表}，按照项目数量缩减复制扩大
    bottom_relationship: 活动紧前关系{活动名称:紧前活动列表}，按照belong中的活动活动ID摘取关系
    return: 缩减后的活动隶属关系和活动紧前关系
    '''
    projects=list(belong.keys())
    new_belong={}
    activities=[]
    need_resource={}
    for i in project_index_set:
        new_belong[projects[i]]=belong[projects[i]]
        activities+=belong[projects[i]]
    
    new_bottom_relationship={}
    for i in activities:
        new_bottom_relationship[i]=bottom_relationship[i]
        need_resource[i]=re_demand[i]
    return new_belong,new_bottom_relationship,activities,need_resource


# 生成一个字典，用于存放bottom_relationship中的活动名称与其索引
def get_ac_index_dict(bottom_relationship):
    ac_index_dict={'start':0,'end':len(bottom_relationship)+1}
    indx=0
    for i in bottom_relationship:
        indx+=1
        ac_index_dict[i]=indx
    
    return ac_index_dict


#定义一个识别项目所有结尾活动的函数
def get_end_activities(belong,relationship):
    '''
    如果不是任何一个活动的紧前活动，那么它就是结尾活动
    belong: 活动隶属关系{项目名称:活动列表}
    relationship: 活动紧前关系{活动名称:紧前活动列表}
    return: 结尾活动{项目名称:活动列表}
    '''
    end_activities={}
    for key,value in belong.items():
        end_activities.update({key:[]})
        for val in value:
            if all(val not in value for value in relationship.values()):
                end_activities[key].append(val)
    return end_activities
#数据预处理
belong_orignal,bottom_relationship_orignal=data_process()
'''
project_index_set=[0]
belong,bottom_relationship,activities,need_resource=adjust_problem_scale(belong,bottom_relationship,project_index_set)

activities.insert(0,'start')
activities.append('end')
ac_index_dict=get_ac_index_dict(bottom_relationship)

plus_relationship=get_plus_relationship(bottom_relationship)
print(f'共有{len(belong)}个项目，共有{len(activities)}个活动')
print(belong)
    #资源集合
resource=list(set(need_resource.values()))

print(f'共有{len(resource)}种资源')
print(resource)
resource_demand=np.zeros((len(activities),len(resource)))
    #活动资源需求,行为活动，列为资源
for i,acti in enumerate(activities):
    if acti =='start'or acti =='end':
        for k,re in enumerate(resource):
            resource_demand[i,k]=capacity[re]#开始和结束活动需要所有的资源
    else:
        re=re_demand[acti]
        k=resource.index(re)
        resource_demand[i,k]=1
    #紧前活动集合
precedence_relationships=[]
for ac,pre_list in plus_relationship.items():
    if pre_list!=[]:
        for pre in pre_list:
            precedence_relationships.append([pre,ac])

print(plus_relationship)
print(f"紧前关系有{len(precedence_relationships)}条")
print(precedence_relationships)
M=999
mu,sigma,d_bar,d_under,dur_time_list=ambigulity(activities)
print(len(mu))
print(len(activities))
data_to_excel(activities,mu,sigma,d_bar,d_under,dur_time_list)

omega=get_omiga(sigma)
print(f'omega={omega}')

e=[1]*len(activities)
end_activities=get_end_activities(belong,bottom_relationship)
print('end_activities')
print(end_activities)'''
#生成路径列表，元素为包含路径组成弧的矩阵
def ge_route(have_re_flow):
    '''
    根据have_re_flow，生成route,have_re_flow=[[None,1,0,0],
                                            [0,None,1,0],
                                            [0,0,None,1],
                                            [0,0,0,None]],为邻接矩阵
    have_re_flow是一个二维列表，表示活动之间的是否有弧连接，第i行第j列为1，表示有活动i指向到活动j的弧,0表示没有弧连接。None表示不能从活动指向本身。
    根据这个二维列表，生成从第一个活动到最后一个活动的所有路径，路径用numpy中的二维数组表示，如果这条路径中包含从i到j的弧，则这个数组的第i行第j列的值为1，否则为0。
    返回一个列表，元素为表示所有路径的二维数组。
    return： X=[array1,array2,array3,...]
    算法逻辑
    采用深度优先搜索DFS直至找到最后一个点，或者没有可以指向的点。其基本思路是：
    V=len(have_re_flow)
    对have_re_flow第一行，遍历每个元素j，如果j=1，生成一个V*V的二维数组，命名为x_j,值全为零，更新x_j第一行的取值，将x_j中j在re_have_flow的位置处的更新为1
        找到当前j的索引index_j,找到have_re_flow的中的index_j行，
        如果全为零，停止搜索
        如果存在1，将首个为1的元素在have_re_flow的位置，在x_j数组中标记为1
        每多一个1，就多一个路径，多生成一个二维数组，命名数按x_1、x_2递增，
        对新生成的二维数组，复制之前x_j除了当前行的所有行，并将当前1在re_have_flow中的位置，在对应新的数组中标记为1
        如果找到的1是最后一列，则停止搜索，否则继续搜索
        重复上述步骤，直至所有路径都被找到
    '''
    V = len(have_re_flow)
    all_paths = []
    
    def dfs(current_node, end_node, visited, current_path):
        # 如果到达终点，保存当前路径
        if current_node == end_node:
            # 将路径转换为邻接矩阵形式
            path_matrix = np.zeros((V, V))
            for i in range(len(current_path)-1):
                path_matrix[current_path[i]][current_path[i+1]] = 1
            all_paths.append(path_matrix)
            return
        
        # 遍历所有可能的下一个节点
        for next_node in range(V):
            # 检查是否有边连接
            if have_re_flow[current_node][next_node] == 1:
                # 添加节点到访问列表和当前路径
                visited.add(next_node)
                current_path.append(next_node)
                # 递归搜索
                dfs(next_node, end_node, visited, current_path)
                # 回溯
                visited.remove(next_node)
                current_path.pop()
    
    # 从起点(0)开始搜索到终点(V-1)
    start_node = 0
    end_node = V-1
    visited = {start_node}
    current_path = [start_node]
    
    # 开始深度优先搜索
    dfs(start_node, end_node, visited, current_path)
    print(f'共有{len(all_paths)}条路径')
    # print(all_paths)
    return all_paths


#生成T和U字典，T存储路径上活动的索引，U存储除去路径上活动之外的活动索引
def get_T_U(all_paths:list)->tuple:
    '''
    all_paths列表中的每个元素是一个矩阵，将矩阵中值为1的行索引位置找出，存在新列表里，表示路径
    返回两个字典，T字典中键为路径索引，值为路径的活动索引列表，U字典中键为路径索引，值为除去路径中活动之外的活动索引列表
    '''
    T = {}  # 存储每条路径上的活动
    U = {}  # 存储不在路径上的活动
    
    # 获取所有活动的索引集合（除去起点和终点）
    V = len(all_paths[0])
    all_activities = set(range(V))
    
    for path_idx, path_matrix in enumerate(all_paths):
        path = []
        # 从起点开始追踪路径
        current_node = 0
        while current_node < V:
            path.append(current_node)
            # 找到当前节点指向的下一个节点
            next_node = -1
            for j in range(V):
                if path_matrix[current_node][j] == 1:
                    next_node = j
                    break
            # 如果没有下一个节点，说明到达终点
            if next_node == -1:
                break
            current_node = next_node
        
        # 将路径存入T字典
        T[path_idx] = path
        # 计算不在路径上的活动并存入U字典
        U[path_idx] = list(all_activities - set(path))
    # print('T字典')
    # print(T)
    # print('U字典')
    # print(U)
    return T, U


#调用子问题求解算法，计算CVaR_loss
def get_CVaR_loss(T:dict,U:dict,all_paths:list,activities:list)->float:
    '''
    计算CVaR_loss
    '''
    I=range(len(activities))
    M=range(len(all_paths))
    gamma=0.99
    
    A=list(belong.keys())
    # omega = 200.0
    e = [1] * len(I)
    cacu_need={}
    
    cacu_need['activities']=activities
    cacu_need['activity_f']=activity_f
    
    try:
        result = solve_optimization_problem(
            A, I, M, T, U, gamma, D_a,cacu_need, mu, sigma, omega, d_bar, d_under, e
        )
        eta, phi, rho=result['eta'],result['phi'],result['rho']
        
        print("子问题最优解：")
        for key, value in result.items():
            print(f"{key}: {value}")
    except OptimizationError as e:
        print(f"错误: {str(e)}")
        result={'objective_value':None,'eta':None,'phi':None,'rho':None}
    CVaR_loss=result['objective_value']
    eta=result['eta']
    phi=result['phi']
    rho=result['rho']
    return CVaR_loss,eta,phi,rho


#主问题求解
def main_pro_solve(cuts:dict)->dict:
    '''
    cuts: 割平面约束集合[{CVaR_loss: float = None,
                                    LB:float=-float('inf'),
                                    set_J:dict}]
    CVaR_loss: 子问题求解结果,例100
    LB: 下界,例0
    set_J: 关键路径字典,例{0: [(0, 2), (2, 4), (4, 5)]
                           1: [(0, 1), (1, 2), (2, 4), (4, 5)]
                           }
    '''

    # Model
    m = gp.Model("main_problem")
    #定义决策变量
    #是否有资源流动
    have_re_flow = []
    '''
    have_re_flow=[[j1,j2,j3,...],
                  [j1,j2,j3,...],
                  [j1,j2,j3,...],
                  ...]
    '''
    for i,ac_i in enumerate(activities):
        have_re_flow.append([])
        for j,ac_j in enumerate(activities):
            if j!=i:
                have_re_flow[i].append(m.addVar(vtype=GRB.BINARY,
                                            name=f"have_re_flow_{ac_i}_{ac_j}"))
            else:
                have_re_flow[i].append(0)
    #资源流动的数量
    number_re_flow=[]
    for i,ac_i in enumerate(activities):
        number_re_flow.append([])
        for j,ac_j in enumerate(activities):
            number_re_flow[i].append([])
            if i!=j:
                for k in resource:
                    number_re_flow[i][j].append(m.addVar(lb=0,vtype=GRB.INTEGER,
                                            name=f"number_re_flow_{ac_i}_{ac_j}_{k}"))
            else:
                number_re_flow[i][j].append(0)
    #弧的最大位置
    location=[]
    for i,ac_i in enumerate(activities):
        location.append([])
        for j in activities:
            if i!=j:
                location[i].append(m.addVar(lb=0,vtype=GRB.INTEGER,
                                        name=f"location_{i}_{j}"))
            else:
                location[i].append(0)
    
    delta=m.addVar(lb=0,vtype=GRB.CONTINUOUS,name='delta')
    
    #约束（2）从每个活动流出的资源正好等于该活动所需要的资源数量
    for i,ac in enumerate(activities):
        for k,re in enumerate(resource):
            #将虚拟的最终活动排除，最后一个活动没有资源流出
            if ac!='end':
                flow_out_sum = gp.quicksum(number_re_flow[i][j][k] 
                                         for j in range(len(activities)) 
                                         if j != i)
                m.addConstr(flow_out_sum == resource_demand[i,k],
                           f"flow_out_{ac}_{re}")
    #约束（3）流进每个活动的资源数刚好等于这个活动需要的资源数量
    for j,ac in enumerate(activities):
        for k,re in enumerate(resource):
            #将虚拟的初始活动排除，初始活动没有资源流进
            if ac!='start':
                flow_in_sum = gp.quicksum(number_re_flow[i][j][k] 
                                        for i in range(len(activities)) 
                                        if i != j)
                m.addConstr(flow_in_sum == resource_demand[j,k],
                           f"flow_in_{ac}_{re}")
    #约束（4）如果两个活动之间存在完成到开始的关系，则资源流不能超过这些活动的最小资源需求
    for k,re in enumerate(resource):
        for i,ac1 in enumerate(activities):
            for j,ac2 in enumerate(activities):
                if i!=j:
                    if [ac1,ac2] not in precedence_relationships:
                        m.addConstr(number_re_flow[i][j][k]<=have_re_flow[i][j]*min(resource_demand[i,k],resource_demand[j,k]),
                                    f"min_redem_{ac1}_{ac2}_{re}")
    
    #约束（5）对于不存在工艺逻辑约束的活动，当没有资源流向时，就没有结束-开始这种活动关系
    for i,ac1 in enumerate(activities):
        for j,ac2 in enumerate(activities):
            if i!=j:
                if [ac1,ac2] not in precedence_relationships:
                    m.addConstr(gp.quicksum(number_re_flow[i][j][k] for k in range(len(resource)))>=have_re_flow[i][j],
                                f"logic_{i}_{j}")
    
    #约束（6）有紧前活动关系则必须有结束-开始关系
    for rel in precedence_relationships:
        pre,after=rel[0],rel[1]
        i,j=ac_index_dict[pre],ac_index_dict[after]
        m.addConstr(have_re_flow[i][j]==1,f"pre_relationship_{rel}")
    #约束（7）没有结束-初始活动这样的关系
    for i,ac in enumerate(activities):
        m.addConstr(have_re_flow[i][0]==0,f"no_activity_into_start_{ac}")
    #约束（8）没有结束-最终活动这样的关系
    for j ,ac in enumerate(activities):
        m.addConstr(have_re_flow[len(activities)-1][j]==0,f"no_activity_out_end{ac}")
    #约束（9）表明从活动i流出的弧的位置必须高于流入活动i的弧的位置。M是最大数。
    for i,ac in enumerate(activities[1:-1]):
        for j,ac2 in enumerate(activities[1:]):
            if i!=j:
                for l in range(len(activities)-1):
                    if l!=i:
                        m.addConstr(location[i][j]>=location[l][i]+1-M*(1-have_re_flow[i][j]),f"location_rank_{ac}_{ac2}")

    #约束条件（10）活动间有结束-开始关系，才可以有弧之间的位置，且规定任何弧的位置不超过n + 1
    for i,ac in enumerate(activities[1:]):
        for j,ac in enumerate(activities[1:]):
            if i!=j:
                m.addConstr(location[i][j]<=(len(activities)-1)*have_re_flow[i][j],
                            f"location_logic_{ac}_{ac2}")
    #约束（11）说明，如果活动0和活动i之间存在finish-to-start关系，则弧（0，i）的位置为1
    for i,ac in enumerate(activities[1:]):
        m.addConstr(location[0][i]==have_re_flow[0][i],f"start1_{i}")
    
    '''约束（12）来自子问题的约束，添加割平面约束'''
    
    #添加之前所有迭代的割平面约束
    for cut in cuts:
        CVaR_loss=cut['CVaR_loss']
        LB=cut['LB']
        set_J=cut['set_J']
        #关键路径可能有多条，添加多个割
        for path_idx,path in set_J.items():
            sum_arc_num_critical_path=0
            for arc in path:
                i,j=arc[0],arc[1]
                sum_arc_num_critical_path+=have_re_flow[i][j]
    
            m.addConstr(delta>=(CVaR_loss-LB)*(sum_arc_num_critical_path-len(path))+CVaR_loss,"cut_constrain")
    
    # Save model
    m.write("RCPSP_main.lp")

    # Solve
    m.setObjective(delta,GRB.MINIMIZE)
    # 启用IIS（不可行子系统）分析
    m.setParam('DualReductions', 0)
    m.optimize()
    # 如果模型不可行，进行诊断
    if m.Status == GRB.INFEASIBLE:
        print("\n模型不可行，正在分析原因...")
        m.computeIIS()
        m.write("model.ilp")
        print("不可行约束已保存到 model.ilp 文件中")
            
        # 获取所有不可行约束
        for c in m.getConstrs():
            if c.IISConstr:
                print(f"不可行约束: {c.ConstrName}")
                    
        return None
    else:
        print(f"\nCVaR_loss: {m.ObjVal:g}")

        # 构建返回结果字典
    result = {
            'status': 'optimal',
            'objective_value': m.ObjVal,
            'variables': {},
            'have_re_flow': [],
            'number_re_flow': [],
            'location': [],
            'delta': None
        }
        
        # 保存所有变量值并按类型分类
    for var in m.getVars():
        result['variables'][var.VarName] = var.X
        if 'have_re_flow' in var.VarName:
            result['have_re_flow'].append(var.X)
        elif 'number_re_flow' in var.VarName:
            result['number_re_flow'].append(var.X)
        elif 'location' in var.VarName:
            result['location'].append(var.X)
        elif var.VarName == 'delta':
            result['delta'] = var.X
    trans_have_re_flow=[]
    values=result['have_re_flow']
    for i,ac in enumerate(activities):
        trans_have_re_flow.append([])
        for j,ac2 in enumerate(activities):
            if i==j:
                trans_have_re_flow[i].append(0)
            else:
                trans_have_re_flow[i].append(values[0])
                values.pop(0)
    result['have_re_flow']=trans_have_re_flow
   
        # 打印结果用于调试
    df = pd.DataFrame({
            "Variable": list(result['variables'].keys()),
            "Value": list(result['variables'].values())
        })
    df.to_excel('output.xlsx')
    print(df)
    # print('主问题输出有向网络')
    # print(result['have_re_flow'])
    print('主问题输出有向网络是否有子回路（是：True，否：False）：')
    print(is_cyclic(result['have_re_flow']))
    return result
    '''if m.Status == GRB.status.OPTIMAL:
        print("模型已成功求解！")
        # Print solution
        print(f"\nCVaR_loss: {m.ObjVal:g}")

        # 构建返回结果字典
        result = {
            'status': 'optimal',
            'objective_value': m.ObjVal,
            'variables': {},
            'have_re_flow': [],
            'number_re_flow': [],
            'location': [],
            'delta': None
        }
        
        # 保存所有变量值并按类型分类
        for var in m.getVars():
            result['variables'][var.VarName] = var.X
            if 'have_re_flow' in var.VarName:
                result['have_re_flow'].append(var.X)
            elif 'number_re_flow' in var.VarName:
                result['number_re_flow'].append(var.X)
            elif 'location' in var.VarName:
                result['location'].append(var.X)
            elif var.VarName == 'delta':
                result['delta'] = var.X

        # 打印结果用于调试
        import pandas as pd
        df = pd.DataFrame({
            "Variable": list(result['variables'].keys()),
            "Value": list(result['variables'].values())
        })
        print(df)
        
        return result
    else:
        print("模型未找到可行解！")
        return {
            'status': 'infeasible',
            'objective_value': None,
            'variables': {},
            'have_re_flow': [],
            'number_re_flow': [],
            'location': [],
            'delta': None
        }'''


#基于Benders分解的求解
def main():
    UB=float('inf')#可行解
    LB=-float('inf')#松弛模型的解
    CVaR_loss=UB
    cuts=[]
    count=0
    num_cut_single_iter=0
    t1=time.time()
    flag='success'
    earliest_start,latest_start,d_value=[],[],[]
    while UB-LB>0.001:
        results=main_pro_solve(cuts)
        #更新下界为\delta
        LB=results['objective_value']
        X=results['have_re_flow']
        from generate_route import remove_redundant_edges,multi_contain_2
        X=remove_redundant_edges(X)
        all_paths=ge_route(X)
        '''
        #找到路径中包含所有项目结尾活动的选择
        contain=[]
        for p in belong.keys():
            for i in end_activities[p]:
                #将所有项目结尾活动的索引加入列表contain
                contain.append(activities.index(i))
        all_paths=multi_contain_2(all_paths,contain)
        '''
        T,U=get_T_U(all_paths)
        
        CVaR_loss,eta,phi,rho=get_CVaR_loss(T,U,all_paths,activities)
        #生成割平面约束
        if CVaR_loss==None:
            print('子问题出错')
            flag='error'
            break
        d_value=solve_d6_problem(eta,phi,mu,d_bar,d_under,rho,e)[1]
        print('d_value')
        print(d_value)
        print('diff_d_mu')
        print([d_value[i]-mu[i] for i in range(len(d_value))])
        #得到多条关键路径，每条关键路径都对应可加入的一个割
        set_P,earliest_start,latest_start=find_critical_paths(X,d_value)
        num_cut_single_iter+=len(set_P)

        cuts.append({'CVaR_loss':CVaR_loss,'LB':LB,'set_J':set_P})
        UB=min(UB,CVaR_loss)
        count+=1
        if count==1:
            first_UB=UB
        print(f'这是第{count}次迭代')
        print(f'当前上界为{UB}')
        print(f'当前下界为{LB}')
        
        print(f'即将添加割平面约束为{len(cuts)}个')
        t2=time.time()
        if count>30 or t2-t1>1800:#迭代次数超过30次或者运行时间超过30分钟
            flag='time_out'
            break
        
    def result_to_excel(earliest_start,latest_start,d_value):
        '''
        将earliest_start,latest_start写入excel文件
        '''
        plan_date=date(2024,1,1)
        def int_to_date(now_date,num):
            return now_date+timedelta(days=num)
        earliest_start_date=[]
        end_time=[]
        for i,time in enumerate(earliest_start):
            earliest_start_date.append(int_to_date(plan_date,time))
            
            end_time.append(int_to_date(plan_date,time+d_value[i]))
        latest_start_date=[]
        for i in latest_start:
            latest_start_date.append(int_to_date(plan_date,i))
        if earliest_start_date==[]:
            earliest_start_date=['']*len(activities)
            latest_start_date=['']*len(activities)
            end_time=['']*len(activities)
        df=pd.DataFrame({'activities':activities,'earliest_start':earliest_start_date,'end_time':end_time})
        df.to_excel(f'output/result_num{len(project_index_set)}_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx',index=False)
        print('schedule_result.xlsx已保存')
    result_to_excel(earliest_start,latest_start,d_value)
    return UB,first_UB,count,num_cut_single_iter/count,flag
if __name__=='__main__':
    zuhe=[]
    # result={}
    projects=['10','11','12','23','29','39','46']
    for idx,project in enumerate(projects):
        zuhe.append(list(range(idx+1)))
    print('-----------zuhe-----------')
    print(zuhe)
    zuhe=[[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[4,5],[0,4,5],[0,1,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5,6]]
    UB_list,first_UB_list,count_list,mean_num_cut_single_iter_list,duration_list=[],[],[],[],[]
    flag_list=[]
    for st in zuhe[2:3]:
        project_index_set=st
        belong,bottom_relationship,activities,need_resource=adjust_problem_scale(belong_orignal,bottom_relationship_orignal,project_index_set)

        activities.insert(0,'start')
        activities.append('end')
        ac_index_dict=get_ac_index_dict(bottom_relationship)

        plus_relationship=get_plus_relationship(bottom_relationship)
        print(f'共有{len(belong)}个项目，共有{len(activities)}个活动')
        print(belong)
            #资源集合
        resource=list(set(need_resource.values()))

        print(f'共有{len(resource)}种资源')
        print(resource)
        resource_demand=np.zeros((len(activities),len(resource)))
            #活动资源需求,行为活动，列为资源
        for i,acti in enumerate(activities):
            if acti =='start'or acti =='end':
                for k,re in enumerate(resource):
                    resource_demand[i,k]=capacity[re]#开始和结束活动需要所有的资源
            else:
                re=re_demand[acti]
                k=resource.index(re)
                resource_demand[i,k]=1
            #紧前活动集合
        precedence_relationships=[]
        for ac,pre_list in plus_relationship.items():
            if pre_list!=[]:
                for pre in pre_list:
                    precedence_relationships.append([pre,ac])

        print(plus_relationship)
        print(f"紧前关系有{len(precedence_relationships)}条")
        print(precedence_relationships)
        M=999
        mu,sigma,d_bar,d_under,dur_time_list=ambigulity(activities)
        print(len(mu))
        print(len(activities))
        data_to_excel(activities,mu,sigma,d_bar,d_under,dur_time_list)

        omega=get_omiga(sigma)
        print(f'omega={omega}')

        e=[1]*len(activities)
        end_activities=get_end_activities(belong,bottom_relationship)
        print('end_activities')
        print(end_activities)
        
        t=time.time()
        UB,first_UB,count,mean_num_cut_single_iter,flag=main()
        UB_list.append(UB)
        first_UB_list.append(first_UB)
        count_list.append(count)
        mean_num_cut_single_iter_list.append(mean_num_cut_single_iter)
        flag_list.append(flag)
        print(f'组合{st}结果为')
        print(UB,first_UB,count,mean_num_cut_single_iter)
        print("相信你自己呀！")
        duration=time.time()-t
        duration_list.append(duration)
        print(f'程序运行时间：{duration}秒')

    df=pd.DataFrame({f'UB{st}':UB_list,'first_UB':first_UB_list,'count':count_list,
                     'mean_cut':mean_num_cut_single_iter_list,'duration':duration_list,
                     'flag':flag_list})
    df.to_excel(f'output/benders_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx',index=False)
        