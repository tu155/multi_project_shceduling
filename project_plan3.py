import copy
import math
import random
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
plt.ion()
from scipy.stats import norm
from math import factorial
from itertools import permutations
from datetime import date,timedelta,datetime
from ast import literal_eval
import time
import transfer_top
import pro_pre_data
import date_process
import wip

from get_exl_data import resource as re_demand,durance as du,relationship_dict as relationship,bom,point,project_acti
from get_exl_data import resource_capacity as capacity,have_worked as worked,due_time,project_status as status
from get_exl_data import resource_calendar


# from pro_pre_data import bottom_relstionship

# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['STSong']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

# plt.rcParams["font.family"] = "FangSong"  # 支持中文显示
def identify_predecessors(relationships, current_activity, critical_path,identified_activities):
    predecessors = relationships.get(current_activity, [])  # 获取当前活动的紧前活动列表

    for activity in predecessors:
        if activity not in critical_path:
            if activity not in identified_activities:
                identified_activities.append(activity)  # 将紧前活动添加到已识别的活动列表中
                # 递归识别紧前活动的紧前活动
                identify_predecessors(relationships, activity,critical_path, identified_activities)
def generate_activity_list(projects, activities):
    """
    生成活动列表，包含每个项目的活动，并为每个活动随机生成优先权。

    Parameters:
    - projects: 项目列表
    - activities: 字典，键是项目名，值是该项目对应的活动列表

    Returns:
    - activity_list: 活动列表，每个活动是一个字典，包含'id'和'priority'字段
    """
    activity_list = [1]

    for project in projects:
        for _ in activities[project]:
            activity_list.append(random.uniform(0, 1))
    activity_list.append(0)

    return activity_list
def date_to_int(now_date,date):
    lenth=(date-now_date).days
    return lenth
def int_to_date(now_date,num):
    return now_date+timedelta(days=num)
class new_project_plan:
    #1.根据订单规模，选择计算初始调度的方法：枚举法/元启发式，注意计划对象为意向订单、已签单未投产订单，已投产订单为资源占用
    #2.根据初始调度计划，逆向调度，找到关键链活动集合和非关键链活动集合
    #3.计算关键链项目缓冲并分配到各项活动，计算非关键链接入缓冲
    #4.将缓冲作为不消耗资源的虚拟活动进行重调度，输出意向订单承诺交期

    def __init__(self,prod_line_calendar,prod_line_status,object_type, size,status,all_successors,have_schedule,have_resource_usage,evaluated_time,projects, activities,relationships,duration,resource_demand,capacity,priorities,
                  num_particles, max_iter, w_max,w_min, c1, c2,num_iterations, tabu_size,now_date,planed_projects):
        self.prod_line_calendar=prod_line_calendar
        self.prod_line_status=prod_line_status
        self.object_type=object_type
        self.size=size
        self.status=status
        self.all_successors=all_successors
        self.have_schedule=have_schedule
        self.have_resource_usage=have_resource_usage
        self.evaluated_time = evaluated_time
        self.projects=list(belong.keys())
        self.activities=activities
        self.relationships=relationships
        self.duration=duration
        self.resource_demand=resource_demand
        self.capacity=capacity
        self.priorities=priorities
        self.num_particles=num_particles
        self.max_iter=max_iter
        self.w_max=w_max
        self.w_min=w_min
        self.c1=c1
        self.c2=c2
        self.num_iterations=num_iterations
        self.tabu_size=tabu_size
        self.now_date=now_date
        self.planed_projects=planed_projects



    def all_acti(self):
        # 增加首尾两点
        # 生成纯活动表、全活动的关系字典
        all_activities = ['start']
        end = set()#end活动的紧前活动初始化为空
        # actis=copy.deepcopy(self.activities)
        all_relationships = {}
        for project in self.projects:
            if self.status[project]==False:
                successors=list(self.all_successors[project])
                for activity in self.activities[project]:
                    if activity not in successors:
                        all_activities.append(activity)
                        all_relationships[activity]=self.relationships[activity]
                        if self.relationships[activity]==[]:
                            all_relationships[activity] = ['start']
            else:       
                for activity in self.activities[project]:
                    all_activities.append(activity)
                    all_relationships[activity]=self.relationships[activity]
                    if self.relationships[activity]==[]:
                        all_relationships[activity] = ['start']
            for activity in all_activities:
                if all(activity not in value for value in all_relationships.values()):
                    end.add(activity)
        all_relationships['end'] = list(end)
        all_activities.append('end')
        # print('----------------')
        # print(all_activities)
        # print(all_relationships)
        return all_activities,all_relationships
        # all_activities=['start', 'A1', 'A2', 'B1', 'B2', 'B3', 'end']
        #all_relationships={'A2': ['A1'], 'B3': ['B1', 'B2'], 'A1': ['start'], 'B1': ['start'], 'B2': ['start'], 'end': ['A2', 'B3']}

    def find_all_feasible_sequence(self):
        activities_list = []
        for project in self.projects:
            for activity in self.activities[project]:
                # if self.status[project]==False and activity in self.all_successors[project]:
                #     continue
                # else:
                activities_list.append(activity)
        # 生成所有可能的排列
        permutations_iterator = permutations(activities_list)
        # 计算排列的总数
        total_permutations = factorial(len(activities_list))
        print('共有' + str(total_permutations) + '初始排列')
        # 存储所有可行列表
        sequence_list = []
        #筛选出符合紧前关系的序列
        for permutation in permutations_iterator:
            # print(permutation)
            is_feasible = []
            for i, activity in enumerate(permutation):
                if activity in self.relationships:
                    if all(value in permutation[:i] for value in self.relationships[permutation[i]]):
                        is_feasible.append(True)
                    else:
                        is_feasible.append(False)
            if all(is_feasible):
                sequence_list.append(list(permutation))
        print('共有' + str(len(sequence_list)) + '种可行的序列，其中前5个为')
        print(sequence_list[:5])
        return sequence_list

    # 删除重复计划
    def is_duplicate(self, new_dict, list_of_dicts):
        for existing_dict in list_of_dicts:
            if all(frozenset(v1.items()) <= frozenset(v2.items()) for k1, v1 in new_dict.items() for k2, v2 in
                   existing_dict.items() if
                   k1 == k2):
                return True
        return False
    #在列表中查找第一个使得lst[i] + 1小于等于target_value的元素，并返回该元素的索引。如果没有找到，返回None
    def find_index(self,lst, start_index, target_value):
        result_index = None
        for i in range(start_index, len(lst)):
            if lst[i] + 1 <= target_value:
                result_index = i
                break
        return result_index

    def date_process(self,now: date, plus: int, calendar: dict):
        '''
        now:当前日期
        plus:要累加的天数
        calendar:工作日历
        return:考虑工作日历之后，累加天数之后的新日期
        '''
        new_date = now
        days_added = 0  # 记录实际增加的天数
        while days_added < plus:  # 循环直到加上所需的天数
            new_date += timedelta(days=1)
            if new_date in calendar and calendar[new_date]:  # 如果新日期是工作日
                days_added += 1  # 增加实际天数
        return new_date

    def binary_search_insert(self,sorted_events, new_event):
        """
        使用二分查找将新的事件元组插入到已排序的事件列表中，保持列表按开始时间升序排列。
        列表中不会出现中间元素开始时间和新事件开始时间相等的情况。

        :param sorted_events: 已排序的事件列表，每个元素是一个(start, end)元组，其中start和end是datetime.date类型。
        :param new_event: 新的事件元组，格式为(start, end)，其中start和end是datetime.date类型。
        :return: 插入新事件后的排序列表。
        """
        new_start = new_event[0]
        left, right = 0, len(sorted_events)

        # 二分查找插入位置
        while left < right:
            mid = (left + right) // 2
            if sorted_events[mid][0] < new_start:
                left = mid + 1
            else:
                right = mid

        # 插入新事件
        sorted_events.insert(left, new_event)
        return sorted_events

    #这是一个具体到设备的项目计划，同时时间颗粒度细化到天，考虑到工作日历
    def get_plan(self,task_seq,new_relationships,task_duration):
        '''
        prod_line_status:生产线占用状态，生产线上空闲和占用时段的分布
        task_seq:任务序列，考虑完紧前关系之后的活动可行序列，包含design,pre,assemble
        prod_line_calendar:生产线工作日历
        task_duration:任务持续时间
        prod_line_capacity:产能，单位：活动个数/天
        '''
        resource_usage=copy.deepcopy(self.have_resource_usage)
        prod_line_status=copy.deepcopy(self.prod_line_status)
        # try_start_time=date(2024,12,30)
        try_start_time=self.now_date
        projects_makespan = {}
        project_max_finish_time = {}
        devation={}
        scheduel_activities = self.have_schedule.copy()
        scheduel_activities['start'] = {'start_time': self.now_date, 'end_time': self.now_date}
        # scheduel_activities={}#计划结果表
        resource_match={}#多工位选择结果表
        # task_duration['end'] = 0
        # self.resource_demand['end']=list(resource_usage.keys())[0]
        for i in task_seq[1:-1]:
            #最早开始时间=所有紧前工序的最晚结束时间
            dependent_end_times = []
            for predecessor in new_relationships[i]:
                dependent_end_times.append(scheduel_activities[predecessor]['end_time'])
                try_start_time = max(dependent_end_times)
            single_duration=task_duration[i]
            resource=self.resource_demand[i]
            #初始化可选时段
            available_periods=[]
            # if i=='2002':
            #     print(i)
            for re_idx,details in prod_line_status[resource].items():
                # 如果设备的大于活动持续时间的时段均为空，那么就跳过这个设备
                # 即只要有任一时段不为空就把这个设备的可行时间纳入available_periods的考虑中。
                if any(i !=[] for i in details[single_duration:]):
                    for j in details[single_duration:]:
                        for p,pair in enumerate(j):
                            if pair[0]>=try_start_time:#时段开始时间晚于尝试开始时间
                                available_periods+=j[p:]#该时段之后的时段均可行
                            elif (pair[1]-try_start_time).days>=single_duration:#时段开始时间早于尝试开始时间，但剩余时间可容纳工时
                                available_periods.append(pair)
            print(i)
            print(single_duration)
            print("available_periods")
            print(available_periods)
            #随机选取可选时段
            choose_period=random.choice(available_periods)
            #根据choose_period识别对应工位与prod_line_status的使用时段
            for re_idx,details in prod_line_status[resource].items():
                for sta_idx,j in enumerate(details[single_duration:]):
                    if choose_period in j :
                        choose_workstation=re_idx#找到选择时段的对应工位，若工位初始状态相同，则索引小的工位工作负荷更大
                        choose_status_idx=sta_idx+single_duration
                        break
                else:#如果内层for循环正常结束，则执行else，即继续外层循环
                    continue
                break#如果内层break，则外层也break
            resource_match[i]=choose_workstation
            #活动开始日期和结束日期
            if choose_period[0]>=try_start_time:
                try_start_time=choose_period[0]
            #计算考虑工作日历后的活动结束日期
            end_date=date_process.date_process(try_start_time,single_duration,self.prod_line_calendar[resource])
            scheduel_activities[i]={}
            scheduel_activities[i]['start_time']=try_start_time
            scheduel_activities[i]['end_time']=end_date
            #更新生产线占用状态
            s_int=date_to_int(self.now_date,try_start_time)
            e_int=date_to_int(self.now_date,end_date)
            for t in range(s_int,e_int):
                n_date=int_to_date(self.now_date,t)
                if self.prod_line_calendar[resource][n_date]:
                    print('t')
                    print(t)
                    resource_usage[resource][t]+=1
            remain=(choose_period[1]-end_date).days#选择时段的剩余空闲时间
            if remain>0:
                #二分法插入对应时段列表
                prod_line_status[resource][choose_workstation][remain]=self.binary_search_insert(prod_line_status[resource][choose_workstation][remain],(end_date,choose_period[1]))
            remain2=(try_start_time-choose_period[0]).days#选择时段的前部分空闲时间
            if remain2>0:
                prod_line_status[resource][choose_workstation][remain2]=self.binary_search_insert(prod_line_status[resource][choose_workstation][remain2],(choose_period[0],try_start_time))
            
            prod_line_status[resource][choose_workstation][choose_status_idx].remove(choose_period)
        
        for project in self.projects:
            max_time = self.now_date
            min_time = self.now_date+timedelta(days=period_length+1)
            # actis=copy.deepcopy(self.activities[project])
            # if self.status[project]==False:
            #     for i in self.all_successors[project]:
            #         actis.remove(i)
            for activity in self.activities[project]:
                if scheduel_activities[activity]['end_time'] > max_time:
                    max_time = scheduel_activities[activity]['end_time']
                if scheduel_activities[activity]['start_time'] < min_time:
                    min_time = scheduel_activities[activity]['start_time']
            projects_makespan[project] = (max_time - min_time).days
            project_max_finish_time[project] = max_time
            if project in self.planed_projects:
                de=(max_time-self.planed_projects[project]).days
                devation[project]=de if de > 0 else 0
        # 使用max()函数和lambda表达式找到值最大的键
        max_key = max(project_max_finish_time, key=lambda k: project_max_finish_time[k])

        scheduel_activities['end'] = {'start_time': project_max_finish_time[max_key], 'end_time': project_max_finish_time[max_key]}
        
        if self.object_type=='min_makespan':
            sum_time = sum(projects_makespan[key] * self.priorities[key] for key in projects_makespan)
        elif self.object_type=='min_max_finish_time':
            sum_time = sum(project_max_finish_time[key] * self.priorities[key] for key in project_max_finish_time)
        elif self.object_type=='min_sum_finish_time':
            sum_time = scheduel_activities['end']['end_time'] - scheduel_activities['start']['start_time']
        elif self.object_type=='min_sum_finish_time_and_devation':
            sum_time = sum(devation[key] for key in devation)
        # sum_time= sum(projects_makespan[key] / priorities[key] for key in projects_makespan)
        print('调度时间表为')
        print(scheduel_activities)
        print('调度总时间及偏差时间之和为')
        print(sum_time)
        print('订单完成时间为')
        print(projects_makespan)
        print(resource_usage)
        return scheduel_activities, sum_time, project_max_finish_time,projects_makespan,resource_usage,resource_match
        
        # return scheduel_activities,resource_match,prod_line_status
    '''
    def get_plan(self,sequence,new_relationships, duration):
        # 1.根据该活动的所有紧前活动最大结束时间设定该活动的预计开始时间，如果没有紧前任务，那么预计开始时间为这个周期的最早时刻
        # 2.检查资源是否满足，如果在预计开始时间该活动所需的资源数量小于等于该资源的剩余数量，并且该活动所需的加工时间小于等于当前周期的剩余时间，
        #   那么把这个活动加入已排程列表，并存储开始时间为预计开始时间，结束时间为预计开始时间+活动持续时间，更新资源剩余数量及周期剩余时间。
        # 3.如果在活动的预计开始时间资源数量小于活动需求的数量，或者该资源在该周期的剩余时间小于活动需求的时间，
        #   那么把这个活动依次试图安排到下一个周期进行，直到需求满足，更新资源数量和时间，并把这个活动加入到已排程列表，存储开始时间和结束时间。
        # 4.将所有活动都加入到已排程列表中时结束，最后输出每个活动的开始时间和结束时间，每个资源在不同周期内的任务开始及结束情况，以及甘特图。
        # 所需数据
        #_,new_relationships=self.all_acti()
        # 初始化数据
        resource_usage=copy.deepcopy(self.have_resource_usage)
        projects_makespan = {}
        project_max_finish_time = {}
        devation={}
        scheduel_activities = self.have_schedule.copy()
        scheduel_activities['start'] = {'start_time': self.date, 'end_time': self.date}
        # scheduel_activities.update(self.have_schedule)
        # have_schedule['have_activity1'] = {'start_time': 0, 'end_time': 1}
        # 依次释放活动(除去start)
        for i, activity in enumerate(sequence[1:]):
            dependent_end_times = []
            # material_demand['end']=[]
            # resource_demand['end']=[]
            duration['end'] = 0
            for predecessor in new_relationships[activity]:

                dependent_end_times.append(scheduel_activities[predecessor]['end_time'])
            try_start_time = max(dependent_end_times)
            # 检查设备可用量
            if activity in self.resource_demand.keys():
                for resource in self.resource_demand[activity].keys():
                    #修改设备使用量的表示方法，{Resource1：[2,1]；resource2:[2,1]}工作中心1的第一时段投产的生产数量为2，第二时段为1，工作中心2同理。
                    #如果设备使用量的列表长度小于准备开始时间，即该设备上的任务还未安排到对应周期，则按工艺和物料采购提前期确定的开始时间可行
                    if len(resource_usage[resource]) < try_start_time:
                            try_end=math.ceil(try_start_time+duration[activity])
                            resource_usage[resource] += [0] * (try_end- len(resource_usage[resource]))
                            resource_usage[resource][try_start_time]+=1
                    #如果任务安排小于当前计划活动的预计结束时间，先将列表长度延长到预计结束时间
                    else:
                        index = self.find_index(resource_usage[resource],try_start_time, self.capacity[resource])
                        if index != None:
                            # print(f"元素值加1后小于容量的第一个元素索引为: {index}")
                            resource_usage[resource][index] += 1
                            try_start_time = index
                        else:
                            # print("在列表中未找到符合条件的元素索引")
                            resource_usage[resource].append(1)
                            try_start_time = len(resource_usage[resource]) - 1
            scheduel_activities[activity] = {'start_time': try_start_time,
                                             'end_time': math.ceil(try_start_time + duration[activity])}
        for project in self.projects:
            max_time = float('-inf')
            min_time = float('inf')
            # actis=copy.deepcopy(self.activities[project])
            # if self.status[project]==False:
            #     for i in self.all_successors[project]:
            #         actis.remove(i)
            for activity in self.activities[project]:
                if scheduel_activities[activity]['end_time'] > max_time:
                    max_time = scheduel_activities[activity]['end_time']
                if scheduel_activities[activity]['start_time'] < min_time:
                    min_time = scheduel_activities[activity]['start_time']
            projects_makespan[project] = max_time - min_time
            project_max_finish_time[project] = max_time
            if project in self.planed_projects:
                devation[project]=max_time-self.planed_projects[project]
        if self.object_type=='min_makespan':
            sum_time = sum(projects_makespan[key] * self.priorities[key] for key in projects_makespan)
        elif self.object_type=='min_max_finish_time':
            sum_time = sum(project_max_finish_time[key] * self.priorities[key] for key in project_max_finish_time)
        elif self.object_type=='min_sum_finish_time':
            sum_time = scheduel_activities['end']['end_time'] - scheduel_activities['start']['start_time']
        elif self.object_type=='min_sum_finish_time_and_devation':
            sum_time = sum(devation[key] for key in devation)+scheduel_activities['end']['end_time'] - scheduel_activities['start']['start_time']
        # sum_time= sum(projects_makespan[key] / priorities[key] for key in projects_makespan)
        # print('调度时间表为')
        # print(scheduel_activities)
        # print('调度总时间为')
        # print(sum_time)
        # print('订单完成时间为')
        # print(projects_makespan)
        # print(resource_usage)
        return scheduel_activities, sum_time, project_max_finish_time,projects_makespan,resource_usage
        # return sum_time
    '''

    def generate_neighbors(self,solution):
        """
        生成当前解的邻域解，这里只是一个简单的例子，实际问题可能需要根据具体情况生成邻域解
        """
        # 生成一个包含五个元素的数组，每个元素都是从均值为零、标准差为0.1的正态分布中独立随机抽样得到的。
        return [solution + np.random.normal(scale=0.1, size=len(solution)) for _ in range(5)]

    # 禁忌搜索算法
    def tabu_search(self, initial_solution,adjacency_matrix):
        """
        Parameters:
        - cost_function: 成本函数，用于评估解的质量
        - initial_solution: 初始解(可更新的编码)
        - num_iterations: 迭代次数
        - tabu_size: 禁忌表的大小

        Returns:
        - best_solution: 最优解
        """
        current_solution = initial_solution.copy()
        best_solution = current_solution.copy()
        tabu_list = []
        all_relationships = self.all_acti()[1]

        for iteration in range(self.num_iterations):
            neighbors = self.generate_neighbors(current_solution)
            feasible_neighbors=[]
            for neighbor in neighbors:
                if not any(np.array_equal(neighbor, arr) for arr in tabu_list):#any()检查可迭代对象中是否至少有一个为True,返回True
                    feasible_neighbors.append(neighbor)
            # feasible_neighbors = [neighbor for neighbor in neighbors if neighbor not in tabu_list]
            feasible_neighbors_sequence=[self.feasible_sequence(feasible_neighbor,adjacency_matrix) for feasible_neighbor in feasible_neighbors]
            neighbors_costs = [self.get_plan(neighbor,all_relationships,self.duration)[1] for neighbor in feasible_neighbors_sequence]

            # 在邻域中选择非禁忌的最佳解
            best_neighbor_idx = np.argmin(neighbors_costs)
            best_neighbor = feasible_neighbors[best_neighbor_idx]
            #best_neighbor_sequence=feasible_neighbors_sequence[best_neighbor_idx]

            # 更新当前解
            current_solution = best_neighbor
            # print('current_solution',current_solution)

            # 更新禁忌表
            tabu_list.append(best_neighbor)
            if len(tabu_list) > self.tabu_size:
                tabu_list.pop(0)

            # 更新全局最佳解
            best_solution_sequence=self.feasible_sequence(best_solution,adjacency_matrix)
            if neighbors_costs[best_neighbor_idx] < self.get_plan(best_solution_sequence,all_relationships,self.duration)[1]:
                best_solution = best_neighbor

        return best_solution

    def adjacency_matrix_generate(self):
        all_activities, all_relationships=self.all_acti()
        # 构建活动到索引的映射
        activity_to_index = {activity: index for index, activity in enumerate(all_activities)}
        # 初始化邻接矩阵
        num_activities = len(all_activities)
        adjacency_matrix = np.zeros((num_activities, num_activities), dtype=int)
        # 填充邻接矩阵
        for successor, predecessors in all_relationships.items():
            successor_index = activity_to_index[successor]
            for predecessor in predecessors:
                predecessor_index = activity_to_index[predecessor]
                adjacency_matrix[predecessor_index, successor_index] = 1
        # 输出邻接矩阵
        # print("Adjacency Matrix:")
        # print(adjacency_matrix)
        return adjacency_matrix

    def feasible_sequence(self,priorities,adjacency_matrix):
        # 已知邻接矩阵，和活动的优先权，
        # 1.先执行邻接矩阵按列汇总和为0的的活动，也就是没有紧前活动的活动，之后把它加入已执行活动列表。
        # 2.按列查看邻接矩阵，如果某个元素的所有紧前活动都在已执行活动列表中，就把他加入到可执行活动列表。
        # 3.根据可执行活动列表中的活动优先权大小，将最大优先权的活动加入到已执行活动列表中。
        # 接着再查看邻接矩阵，也就是回到步骤2.依次进行，直到所有活动都加入到已执行活动列表中。
        #准备所需数据

        all_activities,_=self.all_acti()
        # 初始化数据
        waitting_activities = all_activities.copy()
        

        executed_activities = ['start']+list(self.have_schedule.keys())#已上线加工活动
        for i in executed_activities:
            waitting_activities.remove(i)
        
        while not len(executed_activities) == len(all_activities):
            indices = [all_activities.index(element) for element in waitting_activities if element in all_activities]
            executable_activities = []
            for j in indices:
                executed_number = 0
                indegree = np.sum(adjacency_matrix[:, j])  # 计算j列的入度和
                for i in range(len(all_activities)):
                    if adjacency_matrix[i][j] == 1:
                        if all_activities[i] in executed_activities:
                            executed_number += 1
                if executed_number == indegree:
                    executable_activities.append(all_activities[j])
            max_priority_id = None
            max_priority_value = float('-inf')  # 设置一个负无穷的初始值
            executable_activities_indices = [all_activities.index(element) for element in executable_activities if
                                             element in all_activities]
            # 遍历可执行列表里的活动在all_activities列表里的index对应的活动优先值
            for compare_id in executable_activities_indices:
                if priorities[compare_id] > max_priority_value:
                    max_priority_value = priorities[compare_id]
                    max_priority_id = all_activities[compare_id]#根据id转换成活动名
            executed_activities.append(max_priority_id)
            waitting_activities.remove(max_priority_id)
            # indices = [all_activities.index(element) for element in waitting_activities if element in all_activities]
            # 输出结果
            # print(f"在比较的ID列表中，priority最大的ID是: {max_priority_id}, priority值为: {max_priority_value}")
        # print('可行调度序列为')
        #print(executed_activities)
        for i in list(self.have_schedule.keys()):
            executed_activities.remove(i)
        # for project in self.projects:
        #     if self.status[project]==False:
        #         successors=list(self.all_successors[project])
        #         for i in successors:
        #             executed_activities.remove(i)
        return executed_activities
    # 粒子群优化算法
    def particle_swarm_optimization(self):
        """
        Parameters:
        - cost_function: 成本函数，用于评估解的质量
        - num_particles: 粒子数量
        - num_dimensions: 解的维度
        - max_iter: 最大迭代次数
        - w: 惯性权重
        - c1: 个体认知因子
        - c2: 群体社会因子

        Returns:
        - global_best_position: 最优解的位置
        """
        # 初始化数据
        all_activities,all_relationships=self.all_acti()
        adjacency_matrix = self.adjacency_matrix_generate()
        num_dimensions=len(all_activities)
        
        particles_position = [generate_activity_list(self.projects,self.activities) for _ in range(self.num_particles)]
        particles_velocity = np.zeros((self.num_particles, num_dimensions))  # 初始速度为0
        personal_best_position = particles_position.copy()  # 初始个体最好位置为原始位置
        personal_best_cost = np.array([float('inf')] * self.num_particles)  # 初始个体的最好目标值为正无穷大
        global_best_position = np.zeros(num_dimensions)  # 初始全局最好位置为全0
        global_best_cost = float('inf')  # 初始种群的最好目标值为正无穷大
        target_values = []

        for iteration in range(self.max_iter):
            w = self.w_max - (self.w_max - self.w_min) * iteration / self.max_iter
            for i in range(self.num_particles):
                # best_neighbor = self.tabu_search(particles_position[i],adjacency_matrix)
                best_sequence=self.feasible_sequence(particles_position[i],adjacency_matrix)
                # best_sequence=self.feasible_sequence(best_neighbor,adjacency_matrix)
                print(best_sequence)
                result_tuple = self.get_plan(best_sequence, all_relationships,self.duration)
                cost = result_tuple[1]
                print(result_tuple[0])
                if cost < personal_best_cost[i]:
                    personal_best_cost[i] = cost
                    personal_best_position[i] = particles_position[i]

                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_position = particles_position[i]

                # 更新粒子速度和位置
                r1, r2 = np.random.rand(), np.random.rand()
                # 使用列表推导式进行逐元素相乘
                particles_velocity[i] = [w * v + self.c1 * r1 * (p - x) + self.c2 * r2 * (g - x) for v, p, x, g in
                                         zip(particles_velocity[i], personal_best_position[i], particles_position[i],
                                             global_best_position)]

                # particles_velocity[i] = w * particles_velocity[i] + c1 * r1 * [x - y for x, y in zip(personal_best_position[i] , particles_position[i])] + c2 * r2 * [x - y for x, y in zip(global_best_position , particles_position[i])]
                particles_position[i] = particles_position[i] + particles_velocity[i]
            # best_iteration_tuple = cost_function(projects, activities, relationships, global_best_position, capacity,
            #                                      resource_demand,
            #                                      duration, 8, 0, material_demand, material_capacity, material_leadtime,
            #                                      priority)
            global_best_position = self.tabu_search(particles_position[i],adjacency_matrix)
            target_values.append(global_best_cost)
            # target_values.append(cost)
        iterations = list(range(self.max_iter))

        # 绘制曲线图
        plt.plot(iterations, target_values, marker='o', linestyle='-')

        # 添加标题和标签
        # plt.title('Change of Target Values with Iterations')
        plt.xlabel('迭代次数')
        plt.ylabel('目标函数值')

        # 显示图形
        plt.savefig('迭代图.png')
        global_best_position_sequence=self.feasible_sequence(global_best_position,adjacency_matrix)
        best_sch= self.get_plan(global_best_position_sequence, all_relationships,self.duration)[0]
        print('粒子群循环后的最优编码（活动优先权）为')
        print(global_best_position)
        # 模拟一些数据，例如目标值随迭代次数的变化
        print('计划为')
        print(best_sch)
        return best_sch,target_values[0],target_values[-1]

    # 遍历所有可行序列
    def enumerate_algorithm(self):
        all_feasible_sequence=self.find_all_feasible_sequence()
        _, new_relationships = self.all_acti()
        # 初始化最小目标值为无穷大
        min_makespan = float('inf')
        # best_sch={}
        feasible_list = []
        for i,sequence in enumerate(all_feasible_sequence):
            # print(i)
            # 在遍历过程中添加 'start' 和 'end'
            sequence.insert(0, 'start')  # 在首部添加 'start'
            sequence.append('end')  # 在尾部添加 'end'
            sch, makespan, project_finish_time,project_makespan,resource_usage,material_plan,resource_match = self.get_plan(sequence,new_relationships,self.duration)
            if makespan == min_makespan:
                # 判断是否重复
                if not self.is_duplicate(sch, feasible_list):
                    feasible_list.append(sch)
                #     print("新元素已添加到列表中")
                # else:
                #     print("新元素与列表中的某个元素重复，未添加")
            if makespan < min_makespan:
                feasible_list.clear()
                feasible_list.append(sch)
                min_makespan = makespan
                best_sch=sch
                best_project_finish_time=project_finish_time
                best_resou=resource_usage
                best_ma=material_plan
        print('最优的可行调度列表为')
        print(feasible_list)
        print('首个最优可行调度计划下的资源利用及物料需求为')
        print(best_resou)
        print(best_ma)
        return best_sch, min_makespan, best_project_finish_time,best_resou,best_ma

    def choose_function(self):
        if self.size == "large":
            return self.particle_swarm_optimization()
        elif self.size == "small":
            return self.enumerate_algorithm()[0]
        else:
            return "Invalid size. Please choose 'large' or 'small'."

    def back_find_index(self,lst, start_index, target_value):
        result_index = None
        for i in range(start_index, -1, -1):  # 从 start_index 到 0 的倒序遍历
            if lst[i] + 1 < target_value:
                result_index = i
                break
        return result_index

    def find_critical_chain(self):
        # 1.先对所有活动按照end_time从大到小，将对应的键组成新的列表，依次释放进行逆向调度
        # 2.对每一项活动，初始最晚开始时间=所有紧后工序的最早开始时间==>修改，最初最晚开始时间=紧后工序的最早开始时间-该活动的持续时间
        schedule,first_UB,UB=self.choose_function()
        # plan_max_min=copy.deepcopy(schedule)
        material_plan = {}
        _, all_relationships = self.all_acti()
        resource_usage=copy.deepcopy(self.have_resource_usage)
        prod_line_status=copy.deepcopy(self.prod_line_status)
        resource_match={}#多工位选择结果表
        
        # 创建一个空字典用于存储每个活动的紧后工序
        successors = {}
        # 遍历关系字典，构建紧后工序字典
        for activity, predecessors in all_relationships.items():
            for predecessor in predecessors:
                # 如果紧后工序字典中已经有了对应的键，则将当前活动添加到该键的值中
                if predecessor in successors:
                    successors[predecessor].append(activity)
                # 否则，创建新的键，并将当前活动作为其值的第一个元素
                else:
                    successors[predecessor] = [activity]
        print('紧后活动关系表为')
        print(successors)
        # {'start': ['A1', 'B1', 'B2'], 'A1': ['A2'], 'B1': ['B3'], 'B2': ['B3'], 'A2': ['end'], 'B3': ['end']}
        
        # 按照结束时间从大到小释放
        sorted_keys_by_end_time = sorted(schedule.keys(), key=lambda x: schedule[x]['end_time'], reverse=True)
        # backward_schedual=self.get_plan(sorted_keys_by_end_time,successors,)
        # backward_schedual = copy.deepcopy(schedule)
        # backward_schedual = {activity: {} for activity in schedule.keys()}
        backward_schedual={}
        backward_schedual['end'] = schedule['end']
        self.duration['start'] = 0
        # 初始化数据
        waitting_activities = sorted_keys_by_end_time.copy()
        waitting_activities.remove('start')
        executed_activities = ['end']
        waitting_activities.remove('end')

        start_value=self.now_date+timedelta(period_length+1)#初始化开始活动的开始时间为计划期末
        # indices = [index for index in range(1,len(waitting_activities))]
        while not len(executed_activities) == len(sorted_keys_by_end_time)-1:#start不参与排程
            for activity in waitting_activities:
            # 最晚结束时间=紧后工序的最小开始时间
                min_value = self.now_date+timedelta(days=period_length)  # 结束时间初始化为计划期末
                if all(successor in backward_schedual for successor in successors[activity]):
                    executed_activities.append(activity)
                    waitting_activities.remove(activity)
                    for successor in successors[activity]:
                        if backward_schedual[successor]['start_time'] < min_value:
                            min_value = backward_schedual[successor]['start_time']
                    #最晚结束时间=紧后工序的最早开始时间
                    try_end_time = min_value
                    # try_start_time = math.floor(try_end_time - self.duration[activity])
                    # 检查资源可用量
                    single_duration=self.duration[activity]
                    resource=self.resource_demand[activity]
                    #初始化可选时段
                    available_periods=[]
                    # if i=='2002':
                    #     print(i)
                    for re_idx,details in prod_line_status[resource].items():
                        # 如果设备的大于活动持续时间的时段均为空，那么就跳过这个设备
                        # 即只要有任一时段不为空就把这个设备的可行时间纳入available_periods的考虑中。
                        if any(i !=[] for i in details[single_duration:]):
                            for j in details[single_duration:]:#符合duration要求
                                for p,pair in enumerate(j):
                                    if pair[1]<=try_end_time:
                                        available_periods.append(pair)
                                    elif (try_end_time-pair[0]).days>=single_duration:#时段末时间晚于尝试结束时间，但前部剩余时间可容纳工时
                                        available_periods.append(pair)
                    print(single_duration)
                    print("available_periods")
                    print(available_periods)
                    #随机选取可选时段
                    choose_period=max(available_periods,key=lambda x:x[1])
                    #根据choose_period识别对应工位与prod_line_status的使用时段
                    for re_idx,details in prod_line_status[resource].items():
                        for sta_idx,j in enumerate(details[single_duration:]):
                            if choose_period in j :
                                choose_workstation=re_idx#找到选择时段的对应工位，若工位初始状态相同，则索引小的工位工作负荷更大
                                choose_status_idx=sta_idx+single_duration
                                break
                        else:#如果内层for循环正常结束，则执行else，即继续外层循环
                            continue
                        break#如果内层break，则外层也break
                    resource_match[activity]=choose_workstation
                    #活动开始日期和结束日期
                    # 如果选择的时段中，时段末时间早于尝试结束时间，则更新尝试结束时间为时段末时间
                    if choose_period[1]<try_end_time:
                        try_end_time=choose_period[1]
                    #计算考虑工作日历后的活动结束日期
                    start_date=date_process.back_date_process(try_end_time,single_duration,self.prod_line_calendar[resource])
                    #更新生产线占用状态
                    s_int=date_to_int(self.now_date,start_date)
                    e_int=date_to_int(self.now_date,try_end_time)
                    for t in range(s_int,e_int):
                        n_date=int_to_date(self.now_date,t)
                        if self.prod_line_calendar[resource][n_date]:
                            print('t')
                            print(t)
                            resource_usage[resource][t]+=1
                    remain=(choose_period[1]-try_end_time).days#选择时段的剩余空闲时间
                    if remain>0:
                        #二分法插入对应时段列表
                        prod_line_status[resource][choose_workstation][remain]=self.binary_search_insert(prod_line_status[resource][choose_workstation][remain],(try_end_time,choose_period[1]))
                    remain2=(start_date-choose_period[0]).days#选择时段的前部分空闲时间
                    if remain2>0:
                        prod_line_status[resource][choose_workstation][remain2]=self.binary_search_insert(prod_line_status[resource][choose_workstation][remain2],(choose_period[0],start_date))
                    
                    prod_line_status[resource][choose_workstation][choose_status_idx].remove(choose_period)

                    backward_schedual[activity] = {'start_time': start_date,'end_time': try_end_time}
                    if start_date<start_value:
                        start_value=start_date#更新开始活动的开始时间
                    # plan_max_min[activity]['max_start_time'],plan_max_min[activity]['max_end_time']=backward_schedual[activity]['start_time'],backward_schedual[activity]['end_time']
        
        backward_schedual['start']={'start_time':start_value,'end_time':start_value}
        sum_time = backward_schedual['end']['end_time'] - backward_schedual['start']['start_time']
        print('逆向调度时间表为')
        print(backward_schedual)
        # plan_max_min['end']['max_start_time'], plan_max_min['end']['max_end_time'] =schedule['end']['start_time'],schedule['end']['end_time']
        # df = pd.DataFrame(
        #     [(activity, details['start_time'], details['end_time'],details['max_start_time'],details['max_end_time']) for activity, details in plan_max_min.items()],
        #     columns=['Activity', 'Start Time', 'End Time','max_start_time','max_end_time'])

        # # 写入 Excel 表格
        # df.to_excel(f"plan_min_max{self.projects[0]}.xlsx", index=False)
        # print(resource_usage)
        # print('逆向调度总时间为')
        # print(sum_time)
        # 计算自由时差，识别关键链
        # 初始化数据
        critical_path = []
        non_critical_path = []
        # 判断任务总时差是否为零，分为关键链和非关键链
        slack_time = {}
        for activity in schedule.keys():
            single_slack=(backward_schedual[activity]['start_time']-schedule[activity]['start_time']).days
            if  single_slack<= 0:
                critical_path.append(activity)
            else:
                slack_time[activity] = single_slack
                non_critical_path.append(activity)
        print('slack_time')
        print(slack_time)
        print('关键链和非关键链活动表分别为')
        print(critical_path)
        print(non_critical_path)
        return backward_schedual,resource_usage,material_plan, sum_time, critical_path, non_critical_path, slack_time, successors,schedule,first_UB,UB

    # 计算缓冲量最大值
    def caculate_safe_time(self):
        safe_time = {}
        for activity in self.evaluated_time:
            # 正态分布的期望值和标准差
            mean = (self.evaluated_time[activity][0] + 4 * self.evaluated_time[activity][1] +
                    self.evaluated_time[activity][2]) / 6
            std_dev = (self.evaluated_time[activity][2] - self.evaluated_time[activity][0]) / 6
            # 计算 95% 置信度对应的 Z 分数
            z_score_95 = norm.ppf(0.975)

            # 计算 50% 置信度对应的 Z 分数
            z_score_50 = norm.ppf(0.75)

            # 根据 Z 分数计算对应的值
            value_95 = mean + z_score_95 * std_dev
            value_50 = mean + z_score_50 * std_dev
                # 计算差值
            safe_time[activity] = value_95 - value_50
            # print('safe_time')
            # print(safe_time)
        return safe_time
    def caculate_buffer(self):
        #所需数据
        safe_time=self.caculate_safe_time()
        # material_leadtime_safetime = self.caculate_safe_time(self.material_leadtime_eval_time)
        # resource_buffer = {}
        safe_time['start'] = 0
        safe_time['end'] = 0
        _,_,_,_,critical_path,non_critical_path,slack_time,successors,scheduel,first_UB,UB=self.find_critical_chain()
        all_activities, all_relationships = self.all_acti()
        project_buffer = math.sqrt(sum(value ** 2 for value in [safe_time[activity] for activity in critical_path]))
        # 识别非关键链
        non_critical_chain = {}
        # 除去关键链上的’end‘活动，紧后活动是end不需要设置缓冲，来减少对关键链的影响
        delend = all_relationships.copy()
        del delend['end']
        # 反转关键路径，便于逆向追踪
        reversed_critical_path = critical_path[::-1]
        for activity in reversed_critical_path:
            if activity in delend.keys():
                for predecessor in delend[activity]:
                    if predecessor in non_critical_path:
                        if predecessor != 'start':
                            non_critical_chain[predecessor] = [predecessor]  # 非关键链形如:{'A2':['A2']
                        # non_critical_activities.remove(predecessor)
        # 遍历非关键活动表，识别紧前活动
        for activity in non_critical_chain:
            identify_predecessors(delend, activity, critical_path, non_critical_chain[activity])
        # 打印结果
        print("识别的非关键链列表:", non_critical_chain)
        feeding_buffer = {}
        final_bf = {}
        for j in non_critical_chain:
            feeding_buffer[j] = math.sqrt(
                sum(value ** 2 for value in [safe_time[activity] for activity in non_critical_chain[j]]))
            final_bf[j] = min(feeding_buffer[j], slack_time[j])  # 避免二次资源冲突
        print('项目缓冲，接入缓冲分别为')
        print('project_buffer',project_buffer)
        print('final_bf',final_bf)
        # # 计算物料缓冲
        # for activity in all_activities:
        #     if activity in material_demand:
        #         for material, demand in material_demand[activity].items():
        #             if material in material_leadtime:
        #                 resource_buffer[activity] = material_leadtime_safetime[material]

        return project_buffer, final_bf, non_critical_chain,critical_path,successors,scheduel,first_UB,UB

    #分配项目缓冲
    def allocate_buffer(self):
        # _,_,_,_,critical_path,_,_,successors=self.find_critical_chain()
        all_relationships=self.all_acti()[1]
        project_buffer,final_bf, non_critical_chain,critical_path,successors,scheduel,first_UB,UB=self.caculate_buffer()
        # 数据初始化
        remove_start = critical_path.copy()
        if 'start' in remove_start:
            remove_start.remove('start')
        if 'end' in remove_start:
            remove_start.remove('end')
        activity_priority = {}
        network_complexity = {}
        resource_scarity = {}
        coefficiency = {}
        for activity in remove_start:
            # 活动优先级从所属订单继承
            for project in self.projects:
                if activity in self.activities[project]:
                    activity_priority[activity] = self.priorities[project]
            # 网络复杂度的分子为活动入度与出度之和
            # 计算活动的紧前工序数量之和
            if activity in all_relationships:
                predecessor_number = len(all_relationships[activity])
                # if 'start' in all_relationships[activity]:
                #     predecessor_number -= 1
            else:
                predecessor_number = 0
            # 计算活动的紧后工序数量之和
            if activity in successors:
                successor_number = len(successors[activity])
                # if 'end' in successors[activity]:
                #     successor_number -= 1
            else:
                successor_number = 0
            all_activities_number = sum(len(activity) for activity in self.activities.values())
            network_complexity[activity] = (predecessor_number + successor_number) / all_activities_number
            # 计算资源紧张度
            for all_activity, demand in self.resource_demand.items():
                if activity == all_activity:
                    activity_ratio = self.duration[activity] / self.capacity[demand]
                    resource_scarity[activity] = activity_ratio
            coefficiency[activity] = activity_priority[activity] * network_complexity[activity] * resource_scarity[
                activity]
            # print('综合系数为')
            # print(coefficiency)
        sum_coefficiency = sum(coefficiency.values())
        allocated_buffer_percent = {key: coefficiency[key] / sum_coefficiency for key in coefficiency}
        allocated_buffer = {key: allocated_buffer_percent[key] * project_buffer for key in allocated_buffer_percent}
        print('关键链上各活动分配的缓冲为')
        print(allocated_buffer)
        return allocated_buffer,final_bf,scheduel,first_UB,UB

    def replan_ready(self):
        # 更新活动持续时间，将有缓冲配置的活动加上缓冲量，排程结束后，再将缓冲识别表示
        allocated_buffer,final_bf,scheduel,first_UB,UB=self.allocate_buffer()
        all_activities=scheduel.keys()
        _,all_relationships=self.all_acti()

        # 数据初始化

        new_duration = self.duration.copy()
        for activity in allocated_buffer.keys():
            new_duration[activity] += math.ceil(allocated_buffer[activity])
        for activity in final_bf.keys():
            new_duration[activity] += math.ceil(final_bf[activity])
        print('加入缓冲后，新的活动持续时间')
        print(new_duration)
        return new_duration,allocated_buffer,final_bf,scheduel,first_UB,UB
    def replan(self):
        new_duration,allocated_buffer,final_bf,scheduel,first_UB,UB=self.replan_ready()
        scheduel_activities, sum_time, project_max_finish_time, projects_makespan, resource_usage,resource_match=self.get_plan(list(scheduel.keys()), self.relationships, new_duration)
        #直接写入
        df = pd.DataFrame(
            [(activity, details['start_time'], details['end_time']) for activity, details in scheduel_activities.items()],
            columns=['Activity', 'Start Time', 'End Time']) 
        # 写入 Excel 表格
        df.to_excel(f'output/project_original_pso{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx', index=False)
        
        del scheduel_activities['start']
        del scheduel_activities['end']
        
        up=allocated_buffer|final_bf#合并输入缓冲与关键活动缓冲
        for a_buffer in up:
            scheduel_activities[a_buffer]['end_time']-=timedelta(days=math.ceil(up[a_buffer]))
            scheduel_activities[f'{a_buffer}_buffer']={'start_time':scheduel_activities[a_buffer]['end_time'],'end_time':scheduel_activities[a_buffer]['end_time']+timedelta(days=math.ceil(up[a_buffer]))}
        #结果写入Excel表
        #直接写入
        df = pd.DataFrame(
            [(activity, details['start_time'], details['end_time']) for activity, details in scheduel_activities.items()],
            columns=['Activity', 'Start Time', 'End Time']) 
        # 写入 Excel 表格
        df.to_excel(f'output/project_plan_pso{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx', index=False)
        #按照所属订单分类
        # 创建空字典
        scheduel_project_category = {}

        # 遍历列表中的每个元素，并以元素作为字典名称定义空字典
        for element in self.projects:
            scheduel_project_category[element] = {}

        for key,value in self.activities.items():
            for v in value:
                if v in scheduel_activities:
                    scheduel_project_category[key][v]=scheduel_activities[v]
        activities_and_buffer=list(scheduel_activities.keys())
        
        for activity in activities_and_buffer:
            if 'buffer' in activity:
                for project in self.projects:
                    if activity[:-7] in self.activities[project]:
                        scheduel_project_category[project][activity]=scheduel_activities[activity]
        print('按订单分组之后的计划为')
        print(scheduel_project_category)
        # 创建 Excel 文件
        with pd.ExcelWriter(f'output/project_category_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx') as writer:
            for key, value in scheduel_project_category.items():
                df = pd.DataFrame(value).T  # 转置 DataFrame
                df.index.name = 'Index'  # 设置行索引名称
                df.columns.name = None  # 清除列索引名称
                df.to_excel(writer, sheet_name=key)  # 写入 Excel 文件

        self.have_schedule.update(scheduel_activities)
        self.have_resource_usage=resource_usage
        print('加入缓冲后的计划为')
        print(self.have_schedule)
        print('加入缓冲后的加权总时间为')
        print(sum_time)
        print('订单完成时间为')
        print(project_max_finish_time)
        print('项目工期为')
        print(projects_makespan)
        print('设备使用为')
        print(self.have_resource_usage)
        return self.have_schedule, sum_time, project_max_finish_time, projects_makespan, self.have_resource_usage,scheduel_project_category,resource_match,first_UB,UB

    def pic(self,resource_usage,scheduel_project_category,resource_match):
        # scheduel_activities.update(self.have_schedule)
        # 设置颜色
        # 为每个项目指定随机颜色
        colors = {}
        for project in self.projects:
            # 生成随机颜色，格式为 (R, G, B)，每个通道的值范围为 0 到 1
            color = (random.random(), random.random(), random.random())
            colors[project] = color
        #plt.figure(figsize=(20, 20))
        plt.figure()
        # 绘制甘特图
        for project, activities in scheduel_project_category.items():
            color = colors.get(project, 'gray')
            for activity, details in activities.items():
                start_time = details['start_time']
                end_time = details['end_time']
                if 'buffer' in activity:
                    plt.barh(f'{self.resource_demand[activity[:-7]]}_{resource_match[activity[:-7]]}', end_time - start_time, left=start_time, color=color, alpha=0.5)
                    # plt.text(start_time + (end_time - start_time) / 2, f'{self.resource_demand[activity[:-7]]}_{resource_match[activity[:-7]]}', str(activity), ha='center', va='center', color='black', weight='bold',fontsize=8)
                else:
                    plt.barh(f'{self.resource_demand[activity]}_{resource_match[activity]}', end_time - start_time, left=start_time, color=color)
                    plt.text(start_time + (end_time - start_time) / 2, f'{self.resource_demand[activity]}_{resource_match[activity]}', str(activity), ha='center', va='center', color='black', weight='bold',fontsize=5)
        # 设置y轴为资源名称+工位号
        y_list=[]
        for i in self.capacity:
            for j in range (self.capacity[i]):
                y_list.append(f'{i}_{j}')

        plt.yticks(range(len(y_list[:25])), y_list[:25])
        # # 设置y轴为资源名称+工位号
        # y_list=[]
        # selected_indices = []
        # selected_labels = []
        # index_offset = 0
        # for i in self.capacity:
        #     capacity_value = self.capacity[i]
        #     for j in range(capacity_value):
        #         label = f'资源{i}'
        #         y_list.append(label)
        #         if j == 0:  # 只选取每个资源的第一个工位作为标签
        #             selected_indices.append(index_offset + j)
        #             selected_labels.append(label)
        #     index_offset += capacity_value

        # # 设置纵轴刻度和标签
        # plt.yticks(selected_indices, selected_labels)

        # 设置图形属性
        title_string = '甘特图'
        plt.xlabel('时间')
        plt.ylabel('资源')
        plt.title(title_string)
        # 保存图表为图片文件，可以指定格式如PNG, PDF, SVG等
        plt.savefig(f'甘特图{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        # plt.grid(True)
        # plt.show()

        # 找到值列表的最大长度
        max_length = max(len(lst) for lst in resource_usage.values())

        # 遍历字典中的值列表，并用 0 补齐长度不足的列表
        for key, value in resource_usage.items():
            if len(value) < max_length:
                resource_usage[key] += [0] * (max_length - len(value))
        # 获取资源名称和时期
        resources = list(resource_usage.keys())
        periods = range(len(next(iter(resource_usage.values()))))

        # 创建新的画布用于绘制资源使用率图
        plt.figure(figsize=(10, 6))

        # 绘制每个资源的使用率
        for i, resource in enumerate(resources):
            # 计算使用率
            usage_rate = [usage / self.capacity[resource] * 100 for usage in resource_usage[resource]]
            # 添加柱状图
            plt.bar([x + i * 0.2 for x in periods], usage_rate, width=0.2, label=f'{resource}')

        # 添加标签和标题
        plt.xlabel('时期')
        plt.ylabel('使用率 (%)')
        plt.title('资源使用率')
        # plt.xticks([p + 0.2 for p in periods], periods)  # 设置 x 轴刻度位置
        plt.legend()#显示图例legend图例
        plt.grid(True)
        # 保存图表为图片文件，可以指定格式如PNG, PDF, SVG等
        plt.savefig(f'车间使用率{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png')
        # 显示图表
        # plt.show()
period_length=360#计划周期

if __name__ == '__main__':
    random.seed(20)
#时间范围
    du_range={}
    for i,dur_time in du.items():
        dur_time=du[i]
        dur_time=int(dur_time*random.uniform(1,1.5))
        bound=int(0.8*random.uniform(1,1.25)*dur_time) if int(0.8*random.uniform(1,1.25)*dur_time)>0 else 1
        du_range[i]=[bound,dur_time,int(dur_time+1.2*random.uniform(1,3))]

        # du_range[i]=[dur_time-random.randint(1,3),dur_time,dur_time+random.randint(1,3)]
    
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
    print(belong)
    
    belong_orignal=copy.deepcopy(belong)
    bottom_relationship_orignal=copy.deepcopy(bottom_relationship)
    '''
    
    belong,bottom_relationship,activities,need_resource=adjust_problem_scale(belong,bottom_relationship,[0,1])


    now_date=date(2024,1,1)

    #设置粒子群算法参数
    num_particles =30
    max_iter = 30
    w_max,w_min, c1, c2 = 1.8,0.9, 1.6, 1.8

    # 设置禁忌搜索算法参数
    num_tabu_iterations = 5
    tabu_size = 5

    #随机选取一个设备（根据产能）
    resource_match={key:random.choice(range(capacity[key])) for key in capacity}
    print('resource_match')
    print(resource_match)
    prod_line_status,resource_usage=wip.get_prod_line_status(worked,re_demand,resource_match,capacity,period_length,now_date)
    print('pls')
    print(prod_line_status)

    start=time.time()

    # 实例化
    plan = new_project_plan(resource_calendar,prod_line_status,'min_sum_finish_time_and_devation', 'large',status,res, worked, resource_usage,
                            du_range, projects,
                            belong, bottom_relationship, du, re_demand,
                            capacity,
                            priority, num_particles, max_iter, w_max, w_min, c1, c2,
                            num_tabu_iterations, tabu_size, now_date, due_time)


    updated_schedule, sum_time, project_max_finish_time, projects_makespan, have_resource_usage, scheduel_project_categroy,resource_match,first_UB,UB = plan.replan()
    
    end=time.time()

    elapsed_time = end - start  # 计算运行时间
    print(f"程序运行时间：{elapsed_time}秒")
    plan.pic(have_resource_usage,scheduel_project_categroy,resource_match)
    '''    
    # result=transfer_top.transfer_top(updated_schedule,start_end,laywer,res,status)

    zuhe=[[0],[0,1],[0,1,2],[0,1,2,3],[0,1,2,3,4],[4,5],[0,4,5],[0,1,4,5],[0,1,2,3,4,5],[0,1,2,3,4,5,6]]
    

    UB_list,first_UB_list,count_list,mean_num_cut_single_iter_list,duration_list=[],[],[],[],[]
    for st in zuhe[2:3]:
        project_index_set=st
        from main_prob import adjust_problem_scale
        belong,bottom_relationship,activities,need_resource=adjust_problem_scale(belong_orignal,bottom_relationship_orignal,project_index_set)
        now_date=date(2024,1,1)

        #设置粒子群算法参数
        num_particles =30
        max_iter = 50
        w_max,w_min, c1, c2 = 1.8,0.9, 1.6, 1.8

        # 设置禁忌搜索算法参数
        num_tabu_iterations = 5
        tabu_size = 5

        #随机选取一个设备（根据产能）
        resource_match={key:random.choice(range(capacity[key])) for key in capacity}
        print('resource_match')
        print(resource_match)
        prod_line_status,resource_usage=wip.get_prod_line_status({},re_demand,resource_match,capacity,period_length,now_date)
        print('pls')
        print(prod_line_status)

        start=time.time()

        # 实例化
        plan = new_project_plan(resource_calendar,prod_line_status,'min_sum_finish_time_and_devation', 'large',status,res, {}, resource_usage,
                                du_range, projects,
                                belong, bottom_relationship, du, re_demand,
                                capacity,
                                priority, num_particles, max_iter, w_max, w_min, c1, c2,
                                num_tabu_iterations, tabu_size, now_date, due_time)


        updated_schedule, sum_time, project_max_finish_time, projects_makespan, have_resource_usage, scheduel_project_categroy,resource_match,first_UB,UB = plan.replan()
        UB_list.append(UB)
        first_UB_list.append(first_UB)
        end=time.time()

        elapsed_time = end - start  # 计算运行时间
        print(f"程序运行时间：{elapsed_time}秒")
        duration_list.append(elapsed_time)
        plan.pic(have_resource_usage,scheduel_project_categroy,resource_match)
    df=pd.DataFrame({'UB':UB_list,'first_UB':first_UB_list,'duration':duration_list})
    df.to_excel(f'output/pso_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.xlsx',index=False)