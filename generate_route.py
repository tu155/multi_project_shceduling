'''
Author: tu155 13293356554@163.com
Date: 2025-03-22 14:00:00
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-23 17:14:36
FilePath: \项目计划\generate_route.py
Description: 生成路径列表
'''
import numpy as np
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

def remove_redundant_edges(have_re_flow):
    # 创建邻接矩阵的深拷贝，避免修改原始数据
    matrix = [row.copy() for row in have_re_flow]
    n = len(matrix)
    
    # 遍历所有可能的起点i和终点j（j > i+1）
    for j in range(n):
        predecessors=[]
        for i in range(n):
            if matrix[i][j] == 1:
                predecessors.append(i)
        for i1 in predecessors:
            for i2 in predecessors:
                if matrix[i1][i2]==1:
                    matrix[i1][j]=0

    return matrix

def multi_contain(all_paths,contain):
    '''
    根据have_re_flow，生成route,have_re_flow=[[None,1,0,0],
                                            [0,None,1,0],
                                            [0,0,None,1],
                                            [0,0,0,None]],为邻接矩阵
    bixuan是路径上必须经过的点，是一个列表，如bixuan=[1]表示除了终点，路径还要必须包括索引为1的点。
    return 一个列表，元素为路径，路径用numpy中的二维数组表示，如果这条路径中包含从i到j的弧，则这个数组的第i行第j列的值为1，否则为0。
    '''
    multi_contain_paths=[]
    
    jilu={}#路径：包含的活动列表
    contain_path={i:[] for i in contain}#活动：包含该活动的路径列表
    for j,path in enumerate(all_paths):
        jilu[j]=[]
        flag=True
        for i in contain:
            #如果活动i在路径矩阵中没有流入，即一列均为0，则该路径不包括活动i，flag=False
            if all(value==0 for value in path[:,i]):
                flag=False
            else:
                jilu[j].append(i)
                contain_path[i].append(j)
        if flag:
            multi_contain_paths.append(path)
    def multi_fun(contain_path):
        return 
    def fun(not_have,matrix,m_list):
        primal=matrix
        for n in not_have:
            for p,path in enumerate(contain_path[n]):
                matrix=primal
                # 更新添加路径之后的选择矩阵
                matrix=np.where((matrix + all_paths[path]) == 2, 1, matrix + all_paths[path])
                #更新没有包括的活动
                have_=jilu[path]
                for i in have_:
                    if i in not_have:
                        not_have.remove(i)
                if len(not_have)==0:
                    m_list.append(matrix)
                
                else:
                    fun(not_have,matrix,m_list)
                return m_list
        return m_list
     
    #初始将包含必选活动1的路径作为初始路径
    matrix_list_0=[all_paths[i] for i in contain_path[contain[0]]]
    matrix_list=[]
    for matrix in matrix_list_0:
        new_fea_matrix=fun(contain[1:],matrix,[])
        matrix_list+=new_fea_matrix
                
    all_paths=matrix_list+multi_contain_paths
    print('包含所有必选活动的路径数量：',len(all_paths))
    print(all_paths)
    return all_paths
import numpy as np

def multi_contain_2(all_paths, contain):
    # 预处理每个路径覆盖的必须节点
    jilu = {}
    for idx, path in enumerate(all_paths):
        covered = []
        for i in contain:
            if np.any(path[:, i] != 0):  # 检查节点i是否被该路径覆盖
                covered.append(i)
        jilu[idx] = covered
    
    # 建立每个必须节点对应的路径索引列表
    contain_path = {i: [] for i in contain}
    for idx in jilu:
        for node in jilu[idx]:
            contain_path[node].append(idx)
    
    # 检查是否存在必须节点未被任何路径覆盖
    for node in contain:
        if not contain_path.get(node, []):
            print(f"必须节点 {node} 没有被任何路径覆盖。")
            return []
    
    results = []
    seen = set()
    
    def dfs(remaining, current_matrix, used_paths):
        if not remaining:
            # 去重检查
            key = current_matrix.tobytes()
            if key not in seen:
                seen.add(key)
                results.append(current_matrix.copy())
            return
        # 选择第一个未被覆盖的节点
        next_node = remaining[0]
        # 遍历所有覆盖该节点的路径
        for path_idx in contain_path[next_node]:
            if path_idx in used_paths:
                continue  # 跳过已使用的路径
            path = all_paths[path_idx]
            # 合并路径矩阵
            new_matrix = np.logical_or(current_matrix, path).astype(int)
            # 计算新的未被覆盖节点
            covered = jilu[path_idx]
            new_remaining = [n for n in remaining if n not in covered]
            # 递归调用
            new_used = used_paths.copy()
            new_used.add(path_idx)
            dfs(new_remaining, new_matrix, new_used)
    
    # 初始调用：剩余节点为所有必须节点，当前矩阵全0，未使用任何路径
    dfs(contain.copy(), np.zeros_like(all_paths[0]), set())
    
    # 单独检查直接包含所有必须节点的路径
    for path in all_paths:
        covers_all = True
        for node in contain:
            if not np.any(path[:, node] != 0):
                covers_all = False
                break
        if covers_all:
            key = path.tobytes()
            if key not in seen:
                seen.add(key)
                results.append(path.copy())
    
    print('包含所有必选活动的路径数量：', len(results))
    return results
if __name__ == '__main__':
    have_re_flow=[[None,1,1,1],
                  [0,None,1,1],
                  [0,0,None,1],
                  [0,0,0,None]]
    path_list=ge_route(have_re_flow)
    print(path_list)


    # 示例输入
    have_re_flow = [
        [0, 1, 1, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 0, 0, 0]
    ]

    # 处理并打印结果
    result = remove_redundant_edges(have_re_flow)
    for row in result:
        print(row)
    path_list=ge_route(result)
    print(path_list)
    multi_contain_paths=multi_contain(path_list,contain=[1,2])
    print(multi_contain_paths)
    
    