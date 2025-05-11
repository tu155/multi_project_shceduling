'''
Author: tu155 13293356554@163.com
Date: 2025-03-21 09:28:20
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-24 13:36:34
FilePath: \项目计划\get_critical_path.py
Description: 获取关键路径
'''

def find_critical_paths(have_re_flow, d)->dict:
    """
    计算关键路径的函数。
    
    :param have_re_flow: 邻接矩阵，表示活动之间的依赖关系
    :param d: 活动持续时间列表
    :return: 一个字典，键为关键路径的索引，值为关键路径上的弧（起点活动索引，结束活动索引）
    """
    n = len(d)  # 活动数量
    # 初始化最早开始时间和最晚开始时间
    earliest_start = [0] * n
    latest_start = [0] * n
    
    # 计算最早开始时间
    def calculate_est(have_re_flow, d):
        n = len(d)
        est = [0] * n
    
        
        # 构建每个活动的后驱列表
        successors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if have_re_flow[i][j]==1:
                    successors[i].append(j)
        predecessors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if have_re_flow[i][j]==1:
                    predecessors[j].append(i)
        from collections import deque
        queue = deque()
        for succ in successors[0]:
            queue.append(succ)
        
        while queue:
            current = queue.popleft()
            for pre in predecessors[current]:
                new_est = est[pre] + d[pre]
                if new_est > est[current]:
                    est[current] = new_est
            for succ in successors[current]:
                queue.append(succ)
        
        return est

    earliest_start=calculate_est(have_re_flow,d)
    print('earliest_start:')
    print(earliest_start)

    # 计算最晚开始时间
    # 初始化所有最晚开始时间为最后一个活动的最早开始时间
    for i in range(n):
        latest_start[i]=earliest_start[-1]
    # 最后一个活动的最晚开始时间等于最早开始时间
    
    
    def calculate_lst(have_re_flow, d, last_start):
        n = len(d)
        lst = [float('inf')] * n
        lst[-1] = last_start
        
        # 构建每个活动的后驱列表
        successors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if have_re_flow[i][j]==1:
                    successors[i].append(j)
        predecessors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if have_re_flow[i][j]==1:
                    predecessors[j].append(i)
        from collections import deque
        queue = deque()
        for pred in predecessors[n-1]:
            queue.append(pred)
        
        while queue:
            current = queue.popleft()
            for succ in successors[current]:
                new_lst = lst[succ] - d[current]
                if new_lst < lst[current]:
                    lst[current] = new_lst
            for pred in predecessors[current]:
                queue.append(pred)
        
        return lst

    latest_start=calculate_lst(have_re_flow,d,earliest_start[-1])
    print('latest_start:')
    print(latest_start)
    
    # 找到关键路径上的活动
    critical_paths = []
    for i in range(n):
        for j in range(n):
            if have_re_flow[i][j] == 1 and earliest_start[i] + d[i] >= latest_start[j]:
                critical_paths.append((i, j))
    print('critical_paths:')
    print(critical_paths)
    def find_all_paths(edges, start, end):
        # 构建邻接表
        adj = {}
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            adj[u].append(v)
        
        result = {}
        
        def dfs(current_node, path_edges):
            if current_node == end:
                # 将当前路径添加到结果字典
                index = len(result)
                result[index] = list(path_edges)
                return
            # 如果当前节点没有后继，直接返回
            if current_node not in adj:
                return
            # 遍历所有后继节点
            for next_node in adj[current_node]:
                edge = (current_node, next_node)
                path_edges.append(edge)
                dfs(next_node, path_edges)
                path_edges.pop()  # 回溯，移除最后添加的边
        
        dfs(start, [])
        return result,earliest_start,latest_start

    paths = find_all_paths(critical_paths, 0, n-1)

    '''# 打印结果
    print(paths)
    # 构建关键路径的字典
    critical_paths_dict = {}
    path_index = 0
    while critical_paths:
        path = []
        current = 0  # 从起点开始
        while True:
            for arc in critical_paths[:]:
                if arc[0] == current:
                    path.append(arc)
                    current = arc[1]
                    #critical_paths.remove(arc)
                    break
            else:
                break
        if path:
            critical_paths_dict[path_index] = path
            path_index += 1'''
    print('critical_paths_dict:')
    print(paths)
    return paths

if __name__ == "__main__":

    # 示例用法
    have_re_flow = [
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0]
    ]
    d = [3, 2, 5, 1, 4, 2]
    d=[4.0, 16.0, 4.0, 4.0, 4.0, 11.0, 4.0, 4.0, 22.0, 4.0, 11.0, 8.0, 4.0, 4.0, 4.0]
    have_re_flow=[[0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.0, 1.0, 1.0, 1.0, 1.0, 1.0], 
                  [0.0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0, 1.0, -0.0, 1.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0, 1.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0, -0.0, -0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, -0.0, -0.0, 0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 1.0, -0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0, -0.0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0, 0.0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1.0], 
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0, 0.0], 
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]]
    
    result = find_critical_paths(have_re_flow, d)
    

    





