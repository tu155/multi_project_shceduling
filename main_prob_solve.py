import gurobipy as gp
from gurobipy import GRB
import numpy as np
activities=['start','1','2','3','4','5','6','7','8','9','end']
capacity={'r1':25,'r2':6}
re_demand={'1':{'r1':20,'r2':6},
           '2':{'r1':25,'r2':2},
           '3':{'r1':5,'r2':4},
           '4':{'r1':10,'r2':0},
           '5':{'r1':13,'r2':3},
           '6':{'r1':18,'r2':0},
           '7':{'r1':10,'r2':5},
           '8':{'r1':10,'r2':6},
           '9':{'r1':0,'r2':0}}
bottom_relationship={'start':[],'1':['start'],'2':['1'],'3':['2'],'4':['2'],'5':['3'],'6':['4'],'7':['5','6'],'8':['7'],'9':['8'],'end':['9']}
# 生成一个字典，用于存放bottom_relationship中的活动名称与其索引
ac_index_dict={}

for i in activities:
    ac_index_dict[i]=activities.index(i)
def main_pro_solve(CVaR_loss: float)->dict:
    
    #活动集合
    V=range(len(activities))
    #资源集合
    resource=list(capacity.keys())
    resource_demand=np.zeros((len(activities),len(resource)))
    #活动资源需求,行为活动，列为资源
    for i,acti in enumerate(activities):
        if acti =='start'or acti =='end':
            for k,re in enumerate(resource):
                resource_demand[i,k]=capacity[re]#开始和结束活动需要所有的资源
        else:
            for k,re in enumerate(resource):
                resource_demand[i,k]=re_demand[acti][re]
            #resource_demand[i,k]=re_demand[acti][k]
    #紧前活动集合
    precedence_relationships=[]
    for ac,pre_list in bottom_relationship.items():
        if pre_list!=[]:
            for pre in pre_list:
                precedence_relationships.append([pre,ac])
    print(precedence_relationships)
    M=999

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
                location[i].append(None)
    
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
                                    f"min_redem_{i}_{j}_{k}")
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
        m.addConstr(have_re_flow[i][0]==0,f"no_activity_into_start")
    #约束（8）没有结束-最终活动这样的关系
    for j ,ac in enumerate(activities):
        m.addConstr(have_re_flow[len(activities)-1][j]==0,f"no_activity_out_end")
    #约束（9）表明从活动i流出的弧的位置必须高于流入活动i的弧的位置。M是最大数。
    for i,ac in enumerate(activities[1:-1]):
        for j,ac2 in enumerate(activities[1:]):
            if i!=j:
                for l in range(len(activities)-1):
                    if l!=i:
                        m.addConstr(location[i][j]>=location[l][i]+
                            1-M*(1-have_re_flow[i][j]),f"location_rank_{i}_{j}")

    #约束条件（10）活动间有结束-开始关系，才可以有弧之间的位置，且规定任何弧的位置不超过n + 1
    for i,ac in enumerate(activities[1:]):
        for j,ac in enumerate(activities[1:]):
            if i!=j:
                m.addConstr(location[i][j]<=(len(activities)-1)*have_re_flow[i][j],
                            f"location_logic_{i}_{j}")
    #约束（11）说明，如果活动0和活动i之间存在finish-to-start关系，则弧（0，i）的位置为1
    for i,ac in enumerate(activities[1:]):
        m.addConstr(location[0][i]==have_re_flow[0][i],f"start1_{i}")

    #来自子问题的约束
    m.addConstr(delta>=CVaR_loss,"cut_constrain")
    
    # Save model
    m.write("RCPSP_main.lp")

    # Solve
    m.setObjective(delta,GRB.MINIMIZE)
    
    m.optimize()

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

if __name__=='__main__':
    CVaR_loss=100
    results=main_pro_solve(CVaR_loss)
