import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from data_manager import RCPSPData
from datetime import datetime

class MainProblem:
    def __init__(self, data: RCPSPData):
        self.data = data
        self.model = None
        self.have_re_flow = None
        self.number_re_flow = None
        self.location = None
        self.delta = None
        self.M = 999
        
    def build_model(self):
        """构建主问题模型"""
        self.model = gp.Model("main_problem")
        self._create_variables()
        self._add_constraints()
        
    def _create_variables(self):
        """创建决策变量"""
        # 是否有资源流动
        self.have_re_flow = []
        for i, ac_i in enumerate(self.data.activities):
            self.have_re_flow.append([])
            for j, ac_j in enumerate(self.data.activities):
                if j != i:
                    self.have_re_flow[i].append(
                        self.model.addVar(vtype=GRB.BINARY,
                                        name=f"have_re_flow_{ac_i}_{ac_j}")
                    )
                else:
                    self.have_re_flow[i].append(0)
                    
        # 资源流动的数量
        self.number_re_flow = []
        for i, ac_i in enumerate(self.data.activities):
            self.number_re_flow.append([])
            for j, ac_j in enumerate(self.data.activities):
                self.number_re_flow[i].append([])
                if i != j:
                    for k in self.data.resource_list:
                        self.number_re_flow[i][j].append(
                            self.model.addVar(lb=0, vtype=GRB.INTEGER,
                                            name=f"number_re_flow_{ac_i}_{ac_j}_{k}")
                        )
                else:
                    self.number_re_flow[i][j].append(0)
                    
        # 弧的最大位置
        self.location = []
        for i, ac_i in enumerate(self.data.activities):
            self.location.append([])
            for j in self.data.activities:
                if i != j:
                    self.location[i].append(
                        self.model.addVar(lb=0, vtype=GRB.INTEGER,
                                        name=f"location_{i}_{j}")
                    )
                else:
                    self.location[i].append(None)
                    
        # CVaR变量
        self.delta = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='delta')
        
    def _add_constraints(self):
        """添加约束条件"""
        self._add_flow_conservation_constraints()
        self._add_resource_flow_constraints()
        self._add_precedence_constraints()
        self._add_location_constraints()
        
    def _add_flow_conservation_constraints(self):
        """添加流量守恒约束"""
        # 约束（2）从每个活动流出的资源正好等于该活动所需要的资源数量
        for i, ac in enumerate(self.data.activities):
            for k, re in enumerate(self.data.resource_list):
                if ac != 'end':
                    flow_out_sum = gp.quicksum(
                        self.number_re_flow[i][j][k]
                        for j in range(len(self.data.activities))
                        if j != i
                    )
                    self.model.addConstr(
                        flow_out_sum == self.data.resource_demand_matrix[i,k],
                        f"flow_out_{ac}_{re}"
                    )
                    
        # 约束（3）流进每个活动的资源数刚好等于这个活动需要的资源数量
        for j, ac in enumerate(self.data.activities):
            for k, re in enumerate(self.data.resource_list):
                if ac != 'start':
                    flow_in_sum = gp.quicksum(
                        self.number_re_flow[i][j][k]
                        for i in range(len(self.data.activities))
                        if i != j
                    )
                    self.model.addConstr(
                        flow_in_sum == self.data.resource_demand_matrix[j,k],
                        f"flow_in_{ac}_{re}"
                    )
                    
    def _add_resource_flow_constraints(self):
        """添加资源流约束"""
        # 约束（4）资源流不能超过最小资源需求
        for k, re in enumerate(self.data.resource_list):
            for i, ac1 in enumerate(self.data.activities):
                for j, ac2 in enumerate(self.data.activities):
                    if i != j:
                        if [ac1, ac2] not in self.data.precedence_relationships:
                            self.model.addConstr(
                                self.number_re_flow[i][j][k] <= 
                                self.have_re_flow[i][j] * 
                                self.data.get_min_resource_demand(ac1, ac2, re),
                                f"min_redem_{ac1}_{ac2}_{re}"
                            )
                            
        # 约束（5）资源流存在性约束
        for i, ac1 in enumerate(self.data.activities):
            for j, ac2 in enumerate(self.data.activities):
                if i != j:
                    if [ac1, ac2] not in self.data.precedence_relationships:
                        self.model.addConstr(
                            gp.quicksum(self.number_re_flow[i][j][k] 
                                      for k in range(len(self.data.resource_list))) >= 
                            self.have_re_flow[i][j],
                            f"logic_{i}_{j}"
                        )
                        
    def _add_precedence_constraints(self):
        """添加优先关系约束"""
        # 约束（6）紧前活动关系约束
        for rel in self.data.precedence_relationships:
            pre, after = rel[0], rel[1]
            i, j = self.data.ac_index_dict[pre], self.data.ac_index_dict[after]
            self.model.addConstr(
                self.have_re_flow[i][j] == 1,
                f"pre_relationship_{rel}"
            )
            
        # 约束（7）初始活动约束
        for i, ac in enumerate(self.data.activities):
            self.model.addConstr(
                self.have_re_flow[i][0] == 0,
                f"no_activity_into_start_{ac}"
            )
            
        # 约束（8）结束活动约束
        for j, ac in enumerate(self.data.activities):
            self.model.addConstr(
                self.have_re_flow[len(self.data.activities)-1][j] == 0,
                f"no_activity_out_end_{ac}"
            )
            
    def _add_location_constraints(self):
        """添加位置约束"""
        # 约束（9）位置顺序约束
        for i, ac in enumerate(self.data.activities[1:-1]):
            for j, ac2 in enumerate(self.data.activities[1:]):
                if i != j:
                    for l in range(len(self.data.activities)-1):
                        if l != i:
                            self.model.addConstr(
                                self.location[i][j] >= self.location[l][i] +
                                1 - self.M*(1-self.have_re_flow[i][j]),
                                f"location_rank_{ac}_{ac2}"
                            )
                            
        # 约束（10）位置上界约束
        for i, ac in enumerate(self.data.activities[1:]):
            for j, ac in enumerate(self.data.activities[1:]):
                if i != j:
                    self.model.addConstr(
                        self.location[i][j] <= 
                        (len(self.data.activities)-1)*self.have_re_flow[i][j],
                        f"location_logic_{ac}_{ac2}"
                    )
                    
        # 约束（11）初始位置约束
        for i, ac in enumerate(self.data.activities[1:]):
            self.model.addConstr(
                self.location[0][i] == self.have_re_flow[0][i],
                f"start1_{i}"
            )
            
    def add_cvar_constraint(self, cvar_loss: float):
        """添加CVaR约束"""
        self.model.addConstr(self.delta >= cvar_loss, "cut_constrain")
        
    def solve(self) -> dict:
        """求解模型并返回结果"""
        # 设置目标函数
        self.model.setObjective(self.delta, GRB.MINIMIZE)
        
        # 保存模型
        self.model.write("RCPSP_main.lp")
        
        # 启用IIS（不可行子系统）分析
        self.model.setParam('DualReductions', 0)
        
        # 求解
        self.model.optimize()
        
        # 如果模型不可行，进行诊断
        if self.model.status == GRB.INFEASIBLE:
            print("\n模型不可行，正在分析原因...")
            self.model.computeIIS()
            self.model.write("model.ilp")
            print("不可行约束已保存到 model.ilp 文件中")
            
            # 获取所有不可行约束
            for c in self.model.getConstrs():
                if c.IISConstr:
                    print(f"不可行约束: {c.ConstrName}")
                    
            return None
            
        # 返回结果
        return self._process_results()
    
    def _process_results(self) -> dict:
        """处理求解结果"""
        result = {
            'status': 'optimal',
            'objective_value': self.model.ObjVal,
            'variables': {},
            'have_re_flow': [],
            'number_re_flow': [],
            'location': [],
            'delta': None
        }
        
        # 保存所有变量值
        for var in self.model.getVars():
            result['variables'][var.VarName] = var.X
            if 'have_re_flow' in var.VarName:
                result['have_re_flow'].append(var.X)
            elif 'number_re_flow' in var.VarName:
                result['number_re_flow'].append(var.X)
            elif 'location' in var.VarName:
                result['location'].append(var.X)
            elif var.VarName == 'delta':
                result['delta'] = var.X
                
        # 保存结果到Excel
        self._save_results_to_excel(result)
        
        return result
    
    def _save_results_to_excel(self, result: dict):
        """将结果保存到Excel文件"""
        # 创建DataFrame
        df = pd.DataFrame({
            "Variable": list(result['variables'].keys()),
            "Value": list(result['variables'].values())
        })
        
        # 处理不同类型的变量
        have_re_flow_data = []
        number_re_flow_data = []
        location_data = []
        
        # 处理have_re_flow数据
        for i, ac_i in enumerate(self.data.activities):
            for j, ac_j in enumerate(self.data.activities):
                if i != j:
                    var_name = f"have_re_flow_{ac_i}_{ac_j}"
                    if var_name in result['variables']:
                        have_re_flow_data.append({
                            'From': ac_i,
                            'To': ac_j,
                            'Value': result['variables'][var_name]
                        })
        
        # 处理number_re_flow数据
        for i, ac_i in enumerate(self.data.activities):
            for j, ac_j in enumerate(self.data.activities):
                if i != j:
                    for re in self.data.resource_list:
                        var_name = f"number_re_flow_{ac_i}_{ac_j}_{re}"
                        if var_name in result['variables']:
                            number_re_flow_data.append({
                                'From': ac_i,
                                'To': ac_j,
                                'Resource': re,
                                'Value': result['variables'][var_name]
                            })
        
        # 处理location数据
        for i, ac_i in enumerate(self.data.activities):
            for j, ac_j in enumerate(self.data.activities):
                if i != j:
                    var_name = f"location_{i}_{j}"
                    if var_name in result['variables']:
                        location_data.append({
                            'From': ac_i,
                            'To': ac_j,
                            'Value': result['variables'][var_name]
                        })
        
        # 保存到Excel
        with pd.ExcelWriter(f'optimization_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx') as writer:
            df.to_excel(writer, sheet_name='All Variables', index=False)
            pd.DataFrame(have_re_flow_data).to_excel(
                writer, sheet_name='Resource Flow Existence', index=False
            )
            pd.DataFrame(number_re_flow_data).to_excel(
                writer, sheet_name='Resource Flow Amount', index=False
            )
            pd.DataFrame(location_data).to_excel(
                writer, sheet_name='Arc Locations', index=False
            )
            pd.DataFrame({
                'Metric': ['Objective Value (CVaR_loss)'],
                'Value': [result['objective_value']]
            }).to_excel(writer, sheet_name='Objective Value', index=False) 