# import pro_pre_data
# from pro_pre_data import relationship,bom
# bottom,father,start_end,top,belong,laywer=pro_pre_data.pre_data(relationship,bom)
# from pro_plan import updated_schedule

# updated_schedule = {'a1': {'start_time': 0, 'end_time': 1}, 'a2': {'start_time': 1, 'end_time': 8}, 'd1': {'start_time': 0, 'end_time': 9}, 'b11': {'start_time': 0, 'end_time': 7}, 'b12': {'start_time': 7, 'end_time': 16}, 'b2': {'start_time': 16, 'end_time': 21}, 'd2': {'start_time': 9, 'end_time': 16}, 'e1': {'start_time': 16, 'end_time': 27}, 'e2': {'start_time': 27, 'end_time': 35}}

# all_successors={'p1': {'c1', 'c2'}, 'p2': {'e2', 'd2', 'e1'}}
# status={'p1':False,'p2':True}

def transfer_top(updated_schedule,start_end,level,all_successors,status):
    """
    transfer_top 由底层活动的排程，计算上层活动计划。:
    
    updated_schedule : 底层活动排程
    start_end : 活动所包含下层活动的开始活动与结束活动
    level : 活动层级
    all_successors : 合同生效之后的后续活动
    status : 订单状态
    
    return : 上下层所有活动计划。
    """
    remain=[]
    for p,s in status.items():
        if s==False and p in all_successors:
            remain+=all_successors[p]#所有未生效的订单，合同生效之后的活动，不需要计算
    res = {}
    res.update(updated_schedule)
    sorted_items = sorted(level.items(), key=lambda item: item[1], reverse=True)#按层级降序排列
    # 将排序后的键值对转换回字典
    sorted_dict = dict(sorted_items)

    for key in sorted_dict:
        if key not in remain and key not in res:
            if all( i in res for i in start_end[key]['start_activity']) and all(i in res for i in start_end[key]['end_activity']):
                res[key] = {}
                start_acti_time = []
                for i in start_end[key]['start_activity']:
                    start_acti_time.append(res[i]['start_time'])
                end_acti_time = []
                for i in start_end[key]['end_activity']:
                    end_acti_time.append(res[i]['end_time'])
                res[key]['start_time'] = min(start_acti_time)
                res[key]['end_time'] = max(end_acti_time)
            
    print('final_res')
    print(res)
    return res
# b=transfer_top(updated_schedule,start_end,laywer,all_successors,status)

# a = {'a1': {'start_time': 0, 'end_time': 1}, 'a2': {'start_time': 1, 'end_time': 10},
#      'b11': {'start_time': 0, 'end_time': 8}, 'd1': {'start_time': 0, 'end_time': 8},
#      'd2': {'start_time': 8, 'end_time': 14}, 'e1': {'start_time': 14, 'end_time': 24},
#      'e2': {'start_time': 24, 'end_time': 31}, 'b12': {'start_time': 8, 'end_time': 18},
#      'b2': {'start_time': 18, 'end_time': 24}, 'c1': {'start_time': 24, 'end_time': 33},
#      'c2': {'start_time': 33, 'end_time': 39}, 'b1': {'start_time': 0, 'end_time': 18},
#      'a': {'start_time': 0, 'end_time': 10}, 'b': {'start_time': 0, 'end_time': 24},
#      'c': {'start_time': 24, 'end_time': 39}, 'd': {'start_time': 0, 'end_time': 14},
#      'e': {'start_time': 14, 'end_time': 31}, 'p1': {'start_time': 0, 'end_time': 39},
#      'p2': {'start_time': 0, 'end_time': 31}}
# c={'a1': {'start_time': 0, 'end_time': 1}, 'a2': {'start_time': 1, 'end_time': 8}, 
#    'd1': {'start_time': 0, 'end_time': 9}, 'b11': {'start_time': 0, 'end_time': 7},
#      'b12': {'start_time': 7, 'end_time': 16}, 'b2': {'start_time': 16, 'end_time': 21}, 
#      'd2': {'start_time': 9, 'end_time': 16}, 'e1': {'start_time': 16, 'end_time': 27}, 
#      'e2': {'start_time': 27, 'end_time': 35}, 'b1': {'start_time': 0, 'end_time': 16}, 
#      'a': {'start_time': 0, 'end_time': 8}, 'b': {'start_time': 0, 'end_time': 21}, 
#      'd': {'start_time': 0, 'end_time': 16}, 'e': {'start_time': 16, 'end_time': 35}, 
#      'p2': {'start_time': 0, 'end_time': 35}}

