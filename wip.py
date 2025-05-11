from datetime import date,timedelta,datetime
import random
random.seed(10)
# def to_free_period(prod_line_capacity,schedule):
# prod_line_status={'resource1':{'1-1':
#                                [
#                                 [],[],[],[],[],[],[],[],[(date(2024,12,30),date(2025,1,7))]#索引0代表空闲时长为1的时段分布
#                                 ],#外层列表长度对应这个计划期的天数,内层长度可变，表示外层列表对应长度的空闲时段
#                                '1-2':
#                                [
#                                 [],[],[],[],[],[],[],[],[(date(2024,12,30),date(2025,1,7))]#索引0代表空闲时长为1的时段分布
#                                 ]
#                                 }
#                             }
# resource_usage={'resource1':[]}
# calendar={'resource1':{
#     date(2024, 12, 30): True,
#     date(2024, 12, 31): False,
#     date(2025, 1, 1): False,
#     date(2025, 1, 2): True,
#     date(2025, 1, 3): True,
#     date(2025, 1, 4): True,
#     date(2025, 1, 5): True,
#     date(2025, 1, 6): True,
#     date(2025, 1, 7): True,
# }}

def date_process(now: date, plus: int, calendar: dict):
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

def binary_search_insert(sorted_events, new_event):
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

def date_to_int(now_date,date):
    lenth=(date-now_date).days
    return lenth
def int_to_date(now_date,num):
    return now_date+timedelta(days=num)
#这是一个具体到设备的项目计划，同时时间颗粒度细化到天，考虑到工作日历
def core_fun(prod_line_status,task_seq,new_relationships,prod_line_calendar,task_duration,resource_demand,resource_usage,now_date):
    '''
    prod_line_status:生产线占用状态，生产线上空闲和占用时段的分布
    task_seq:任务序列，考虑完紧前关系之后的活动可行序列，包含design,pre,assemble
    prod_line_calendar:生产线工作日历
    task_duration:任务持续时间
    prod_line_capacity:产能，单位：活动个数/天
    '''
    for i in resource_usage:
        if len(resource_usage[i])<10:
            resource_usage[i]+=[0]*(10-len(resource_usage[i]))
    try_start_time=now_date
    scheduel_activities={}#计划结果表
    resource_match={}#多工位选择结果表
    for i in task_seq:
        dependent_end_times = []
        for predecessor in new_relationships[i]:
            dependent_end_times.append(scheduel_activities[predecessor]['end_time'])
            try_start_time = max(dependent_end_times)
        single_duration=task_duration[i]
        resource=resource_demand[i]
        if i=='2317':
            print(f'2317对应{prod_line_status[resource]}')
        available_periods=[]
        for re_idx,details in prod_line_status[resource].items():
            for j in details[single_duration:]:
                for p,pair in enumerate(j):
                    if pair[0]>=try_start_time:#时段开始时间晚于尝试开始时间
                        available_periods+=j[p:]#该时段之后的时段均可行
                    elif (pair[1]-try_start_time).days>=single_duration:#时段开始时间早于尝试开始时间，但剩余时间可容纳工时
                        available_periods.append(pair)
        choose_period=random.choice(available_periods)#随机选取可选时段
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
        end_date=date_process(try_start_time,single_duration,prod_line_calendar[resource])
        scheduel_activities[i]={}
        scheduel_activities[i]['start_time']=try_start_time
        scheduel_activities[i]['end_time']=end_date
        
        #更新生产线占用状态
        s_int=date_to_int(now_date,try_start_time)
        e_int=date_to_int(now_date,end_date)
        for t in range(s_int,e_int):
            n_date=int_to_date(now_date,t)
            if prod_line_calendar[resource][n_date]:
                resource_usage[resource][t]+=1
        remain=(choose_period[1]-end_date).days#选择时段的后半部分剩余空闲时间
        if remain>0:
            #二分法插入对应时段列表
            prod_line_status[resource][choose_workstation][remain-1]=binary_search_insert(prod_line_status[resource][choose_workstation][remain],(end_date,choose_period[1]))
        remain2=(try_start_time-choose_period[0]).days#选择时段的前部分空闲时间
        if remain2>0:
            prod_line_status[resource][choose_workstation][remain2-1]=binary_search_insert(prod_line_status[resource][choose_workstation][remain2],(choose_period[0],try_start_time))
        
        prod_line_status[resource][choose_workstation][choose_status_idx].remove(choose_period)
    return scheduel_activities,resource_match,prod_line_status,resource_usage

# a=core_fun(prod_line_status,['a1','a2'],{'a1':[],'a2':['a1']},calendar,{'a1':2,'a2':3},{'a1':'resource1','a2':'resource1'},resource_usage)
# print(a)
# '''
# ({'a1': {'start_time': datetime.date(2024, 12, 30), 'end_time': datetime.date(2025, 1, 3)}, 'a2': {'start_time': datetime.date(2025, 1, 3), 'end_time': datetime.date(2025, 1, 6)}}, 
#  {'a1': '1-1', 'a2': '1-2'}, 
#  {'resource1': {'1-1': [[], [], [], [], [(datetime.date(2025, 1, 3), datetime.date(2025, 1, 7))], [], [], [], []], 
#                 '1-2': [[], [(datetime.date(2025, 1, 6), datetime.date(2025, 1, 7))], [], [], [(datetime.date(2024, 12, 30), datetime.date(2025, 1, 3))], [], [], [], []]}})
# '''
def get_prod_line_status(schedule_activities,resource_demand,resource_match,resource_capacity,period_length,now_date):
    prod_line_status={}
    re_id_busy={}
    #初始化设备使用
    resource_usage={}
    for i in resource_capacity:
        resource_usage[i]=[0]*(period_length+1)#40天，1号到5号有4天，因为左闭右开
    
    #初始化生产线状态
    for i,capa in resource_capacity.items():
        prod_line_status[i]={}
        for j in range(capa):
            re_id_busy[(i,j)]=[]
            prod_line_status[i][j]=[[] for _ in range(period_length-1)]#初始化n-1个空闲时段为空
            prod_line_status[i][j].append([(now_date,now_date+timedelta(days=period_length))])#最后一个空闲时段为计划制定时间到计划周期末
    #根据现有任务修改生产线资源占用
    for acti,value in schedule_activities.items():
        resource=resource_demand[acti]
        s_int=date_to_int(now_date,datetime.strptime(value['start_time'],"%Y-%m-%d").date())
        e_int=date_to_int(now_date,datetime.strptime(value['end_time'],"%Y-%m-%d").date())
        for t in range(s_int,e_int):
            n_date=int_to_date(now_date,t)
            if prod_line_status[resource][n_date]:
                resource_usage[resource][t]+=1
    #根据现有任务修改生产线状态
    for a,sche in schedule_activities.items():
        i=resource_demand[a]
        j=resource_match[a]
        re_id_busy[(i,j)].append((sche['start_time'],sche['end_time']))
    for idx,value in re_id_busy.items():
        if value !=[]:
            prod_line_status[idx[0]][idx[1]][-1]=[]#将最长空闲时段删除
            sorted_value=sorted(value,key=lambda x:x[0])#按照开始时间升序排列
            free_periods=find_free_periods(now_date,now_date+timedelta(days=period_length),sorted_value)
            for fp in free_periods:
                differ=(fp[1]-fp[0]).days
                prod_line_status[idx[0]][idx[1]][differ-1]=binary_search_insert(prod_line_status[idx[0]][idx[1]][differ-1],fp)
    return prod_line_status,resource_usage


def find_free_periods(start_date, end_date, occupied_periods):
    # 初始化空闲时段列表
    free_periods = []
    
    # 定义一个变量来存储当前检查的时间点
    current_date = start_date
    
    # 遍历占用时间段
    for occupied_start, occupied_end in occupied_periods:
        # 如果当前日期在占用时间段之前，添加空闲时段
        if current_date < occupied_start:
            free_periods.append((current_date, occupied_start))
        # 更新当前日期为占用时间段的结束日期的下一天
        current_date = occupied_end
    
    # 如果最后一个占用时间段之后还有时间，添加最后的空闲时段
    if current_date <=end_date:
        free_periods.append((current_date, end_date))
    
    return free_periods

# # 示例使用
# start_date = date(2024, 12, 1)  # 起始日期 2024-12-05
# end_date = date(2024, 12, 10)    # 结束日期 2024-12-07
# occupied = [
#     (date(2024, 11, 30), date(2024, 12, 6)),
#     (date(2024, 12, 8), date(2024, 12, 9))
# ]

# # 调用函数并打印结果
# free_times = find_free_periods(start_date, end_date, occupied)
# for period in free_times:
#     print(f"Free period from {period[0]} to {period[1]}")

# prod_line_status,resource_usage=get_prod_line_status({'a1': {'start_time': date(2024, 12, 30), 'end_time':date(2025, 1, 3)}, 'a2': {'start_time': date(2025, 1, 3), 'end_time': date(2025, 1, 6)}}, 
# {'a1':'resource1','a2':'resource1'},{'a1': 0, 'a2': 1}, {'resource1':2},10,date(2024, 12, 30))
# print(prod_line_status,resource_usage)
# '''
# {'resource1': {0: [[], [], [], [], [], [(datetime.date(2025, 1, 3), datetime.date(2025, 1, 9))], [], [], [], []], 
#                 1: [[], [], [(datetime.date(2025, 1, 6), datetime.date(2025, 1, 9))], [(datetime.date(2024, 12, 30), datetime.date(2025, 1, 3))], [], [], [], [], [], []]}}
# '''