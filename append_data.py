'''
Author: tu155 13293356554@163.com
Date: 2025-03-28 21:19:39
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-29 10:30:00
FilePath: \项目计划\append_data.py
Description: 补充D、b_f,w_f的值
'''
from get_exl_data import due_time
from datetime import datetime, date
import unittest

projects = ['10', '11', '12', '23', '29', '39', '46']


def cacu_differ_d(plan_date: date, due_date: date):
    # 计算两个日期之间的天数差异
    differ_d = (due_date - plan_date).days
    return differ_d


D_a = {}
for i in projects:
    due_date = due_time[i]
    print(due_date)
    print(type(due_date))
    plan_date = date(2024, 11, 1)
    differ_d = cacu_differ_d(plan_date, due_date)
    print(differ_d)
    D_a[i] = differ_d
print(D_a)


def confirm_calender():
    b_f = {}
    w_f = {}
    # key=1,制造
    b_f[1] = 6
    b_f[2] = 5

    w_f[1] = 1
    w_f[2] = 2

    return b_f, w_f


b_f, w_f = confirm_calender()
print(b_f)
print(w_f)

class cacu_n_laf():
    def __init__(self,adj_matrix,a_set,end_activities,activities,activity_f):
        self.adj_matrix=adj_matrix
        self.a_set=a_set
        self.end_activities=end_activities
        self.adj_matrix=adj_matrix
        self.a_set=a_set
        self.end_activities=end_activities
        self.activities=activities
        self.activity_f=activity_f#活动类型
        
    def get_path_indices(self):
        """
        输入邻接矩阵，返回从起点（索引 0）到终点（最后一个索引）经过的索引列表。
        如果没有路径，则返回空列表。

        :param adj_matrix: 邻接矩阵，二维列表
        :return: 从起点到终点经过的索引列表
        """
        V = len(self.adj_matrix)
        start_node = 0
        end_node = V - 1
        found_path = []

        def dfs(current_node, path):
            nonlocal found_path
            if current_node == end_node:
                found_path = path[:]
                return
            for next_node in range(V):
                if self.adj_matrix[current_node][next_node] == 1 and next_node not in path:
                    path.append(next_node)
                    dfs(next_node, path)
                    if found_path:
                        return
                    path.pop()

        dfs(start_node, [start_node])
        return found_path


    class TestGetPathIndices(unittest.TestCase):
        def test_has_path(self):
            # 有路径的邻接矩阵
            adj_matrix = [
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ]
            expected = [0, 1, 2]
            result = self.get_path_indices(adj_matrix)
            self.assertEqual(result, expected)

        def test_no_path(self):
            # 无路径的邻接矩阵
            adj_matrix = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
            expected = []
            result = self.get_path_indices(adj_matrix)
            self.assertEqual(result, expected)

        def test_single_node(self):
            # 只有一个节点的邻接矩阵
            adj_matrix = [[0]]
            expected = [0]
            result = self.get_path_indices(adj_matrix)
            self.assertEqual(result, expected)

    def a_pre(self, path, a):
        '''
        输入：
            path:路径[1,2,3,4,5,6,7,8,9,10]
            end_activities:项目的结束活动{1:[8,9]，2:[10]}
            a:项目
        输出：路径上a的前驱活动列表，返回 idx 最大的前序活动列表，如果没有则返回空列表
        '''
        max_idx = -1
        a_pre_list = []
        if a in self.end_activities:
            for i in self.end_activities[a]:
                if i in path:
                    idx = path.index(i)
                    if idx > max_idx:
                        max_idx = idx
                        a_pre_list = path[:idx + 1]
        return a_pre_list
    def get_n_laf(self,a_pre_list):
        num_f1=0
        num_f2=0
        for idx in a_pre_list:
            activity=self.activities[idx]
            f=self.activity_f[activity]
            if f=='MAKE':
                num_f1+=1
            else:
                num_f2+=1
        return num_f1,num_f2
    def main(self):
        '''
        输入：
            adj_matrix:邻接矩阵
            a_set:项目集合
        输出：
            n_laf:项目的n_laf值
        '''
        a_nlaf={}
        for a in self.a_set:
            path=self.get_path_indices()
            a_pre_list=self.a_pre(path,a)
            num_f1,num_f2=self.get_n_laf(a_pre_list)

            a_nlaf[a]=[num_f1,num_f2]
        return a_nlaf

if __name__ == '__main__':
    
            # 初始化测试数据
    adj_matrix = [
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0]
        ]
    a_set = {1, 2}
    end_activities = {1: [1], 2: [2]}
    activities = [0, 1, 2]
    activity_f = {0: 'OTHER', 1: 'MAKE', 2: 'OTHER'}

    cacu = cacu_n_laf(adj_matrix, a_set, end_activities, activities, activity_f)
    n_=cacu.main()