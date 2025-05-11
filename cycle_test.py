'''
Author: tu155 13293356554@163.com
Date: 2025-03-29 12:09:39
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-29 12:11:30
FilePath: \项目计划\cycle_test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# ... 已有代码 ...

def is_cyclic_util(v, visited, rec_stack, graph):
    """
    辅助函数，用于递归检查是否存在环
    """
    # 标记当前节点为已访问，并加入递归栈
    visited[v] = True
    rec_stack[v] = True

    # 遍历所有相邻节点
    for neighbor in range(len(graph)):
        if graph[v][neighbor] == 1:
            if not visited[neighbor]:
                if is_cyclic_util(neighbor, visited, rec_stack, graph):
                    return True
            elif rec_stack[neighbor]:
                return True

    # 从递归栈中移除当前节点
    rec_stack[v] = False
    return False

def is_cyclic(graph):
    """
    检查邻接矩阵表示的图是否存在环
    """
    num_vertices = len(graph)
    visited = [False] * num_vertices
    rec_stack = [False] * num_vertices

    for node in range(num_vertices):
        if not visited[node]:
            if is_cyclic_util(node, visited, rec_stack, graph):
                return True

    return False

def main():
    graph_without_cycle = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
    graph_with_cycle = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0]
        ]
    print("Graph without cycle:", is_cyclic(graph_without_cycle))  # 输出: False
    print("Graph with cycle:", is_cyclic(graph_with_cycle))  # 输出: True
if __name__ == "__main__":
    main()

# ... 已有代码 ...
