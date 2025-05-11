import numpy as np
import main_prob

class RCPSPData:
    def __init__(self):
        # 基础数据
        self.activities = main_prob.activities
        #['start','1','2','3','4','5','6','7','8','9','end']
        self.capacity = main_prob.capacity
        
        self.bottom_relationship = main_prob.bottom_relationship
        
        # 处理后的数据
        self.ac_index_dict = main_prob.ac_index_dict
        self.resource_list = main_prob.resource
        self.precedence_relationships = main_prob.precedence_relationships
        self.resource_demand_matrix = main_prob.resource_demand
        
        # 输出初始化数据
        self.print_debug_info()
        
    def print_debug_info(self):
        """打印调试信息"""
        print("\n=== 数据验证信息 ===")
        print("\n1. 活动列表:")
        print(self.activities)
        
        print("\n2. 资源列表:")
        print(self.resource_list)
        
        print("\n3. 资源容量:")
        for res in self.resource_list:
            print(f"{res}: {self.capacity[res]}")
        
        print("\n4. 资源需求矩阵:")
        print("活动/资源", end="\t")
        for res in self.resource_list:
            print(f"{res}", end="\t")
        print()
        for i, act in enumerate(self.activities):
            print(f"{act}", end="\t\t")
            for k in range(len(self.resource_list)):
                print(f"{self.resource_demand_matrix[i,k]}", end="\t")
            print()
            
        print("\n5. 活动索引字典:")
        print(self.ac_index_dict)
            
        print("\n6. 优先关系:")
        for act, preds in self.bottom_relationship.items():
            if preds:  # 只打印有前导活动的活动
                print(f"{act}: {preds}")
                
        print("\n7. 优先关系列表形式:")
        for rel in self.precedence_relationships:
            print(f"{rel[0]} -> {rel[1]}")
    
    def _create_activity_index_dict(self) -> dict:
        """创建活动索引字典"""
        return {i: idx for idx, i in enumerate(self.activities)}
    
    def _create_precedence_relationships(self) -> list:
        """创建紧前活动关系列表"""
        relationships = []
        for ac, pre_list in self.bottom_relationship.items():
            if pre_list:
                for pre in pre_list:
                    relationships.append([pre, ac])
        return relationships
    
    def _create_resource_demand_matrix(self) -> np.ndarray:
        """创建资源需求矩阵"""
        resource_demand = np.zeros((len(self.activities), len(self.resource_list)))
        for i, acti in enumerate(self.activities):
            if acti == 'start' or acti == 'end':
                for k, re in enumerate(self.resource_list):
                    resource_demand[i,k] = self.capacity[re]
            else:
                for k, re in enumerate(self.resource_list):
                    resource_demand[i,k] = self.re_demand[acti][re]
        return resource_demand
    
    def get_activity_count(self) -> int:
        """获取活动数量"""
        return len(self.activities)
    
    def get_resource_count(self) -> int:
        """获取资源类型数量"""
        return len(self.resource_list)
    
    def get_min_resource_demand(self, activity1: str, activity2: str, resource: str) -> float:
        """获取两个活动之间的最小资源需求"""
        i1 = self.ac_index_dict[activity1]
        i2 = self.ac_index_dict[activity2]
        k = self.resource_list.index(resource)
        return min(self.resource_demand_matrix[i1,k], self.resource_demand_matrix[i2,k])
    
    def validate_data(self) -> bool:
        """验证数据的合法性"""
        try:
            # 验证活动列表
            if not self.activities or len(self.activities) < 3:  # 至少需要start、end和一个实际活动
                print("错误：活动列表为空或活动数量过少")
                return False
                
            # 验证资源容量
            if not self.capacity:
                print("错误：未定义资源容量")
                return False
                       
            # 验证优先关系
            for act, preds in self.bottom_relationship.items():
                if act not in self.activities:
                    print(f"错误：优先关系中的活动 {act} 不在活动列表中")
                    return False
                for pred in preds:
                    if pred not in self.activities:
                        print(f"错误：活动 {act} 的前导活动 {pred} 不在活动列表中")
                        return False
            
            # 验证是否存在循环依赖
            if not self._check_no_cycles():
                print("错误：优先关系中存在循环依赖")
                return False
            
            return True
            
        except Exception as e:
            print(f"数据验证过程中发生错误：{str(e)}")
            return False
            
    def _check_no_cycles(self) -> bool:
        """检查优先关系是否存在循环依赖"""
        def dfs(node, visited, path):
            visited.add(node)
            path.add(node)
            
            for neighbor in self.bottom_relationship.get(node, []):
                if neighbor not in visited:
                    if not dfs(neighbor, visited, path):
                        return False
                elif neighbor in path:
                    return False
                    
            path.remove(node)
            return True
            
        visited = set()
        path = set()
        
        for node in self.activities:
            if node not in visited:
                if not dfs(node, visited, path):
                    return False
        return True 