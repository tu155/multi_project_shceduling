# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
'''relationship={'p1':[],
    'a':[],
   'b':[],
   'c':['a','b'],
   'a1':[],
   'a2':['a1'],
   'b1':[],
   'b2':['b1'],
   'c1':[],
   'c2':['c1'],
   'b11':[],
   'b12':['b11'],
              'p2':[],
              'd':[],
              'e':['d'],
              'd1':[],
              'd2':['d1'],
              'e1':[],
              'e2':['e1']}
bom={'p1':['a','b','c'],
    'a':['a1','a2'],
     'b':['b1','b2'],
     'c':['c1','c2'],
     'b1':['b11','b12'],
                'p2':['d','e'],
                'd':['d1','d2'],
                'e':['e1','e2']}
status={'p1':False,'p2':True}#项目1还没有合同生效,p2合同生效
point={'p1':'c1','p2':'d2'}#不同项目中代表合同生效的活动id
'''
from get_exl_data import project_status as status,point,bom,relationship_dict as relationship
def pre_data(relationship,bom):

    #父件
    father={}
    for key,value in bom.items():
        for v in value:
            father[v]=key
    print('father')
    print(father)
    #紧后活动
    next_acti={}
    for key,value in relationship.items():
        if len(value)!=0:
            for v in value:
                if v in next_acti:
                    next_acti[v].append(key)
                else:
                    next_acti[v]=[key]
    for key in relationship:
        if key not in next_acti:
            next_acti[key]=[]
    print('next_acti')
    print(next_acti)

    #识别底层活动列表
    bottom=[]
    for i in relationship:
        if i not in bom:
            bottom.append(i)
    print('bottom')
    print(bottom)
    #最顶层活动
    middle=[]
    for key,value in bom.items():
        for v in value:
            middle.append(v)
    top=[t for t in relationship if t not in middle]
    print('top')
    print(top)
    
    #首尾活动识别
    start_end={}
    for tm,sub in bom.items():
        start_end[tm]={'start_activity':[],'end_activity':[]}
        for su in sub:
            if relationship[su]==[] or all(pre not in sub for pre in relationship[su]):
                start_end[tm]['start_activity'].append(su)
            if next_acti[su]==[] or all(net not in sub for net in next_acti[su]):
                start_end[tm]['end_activity'].append(su)
    print('start_end')
    print(start_end)
    #底层活动归属到项目
    belong={fa:[] for fa in top}
    for i in bottom:
        fa=father[i]
        while fa not in top:
            fa=father[fa]
        belong[fa].append(i)
    print('belong')
    print(belong)
    #层级
    level={}
    for i in top:
        level[i]=0
    for key,value in bom.items():
        if key in level:
            for v in value:
                level[v]=level[key]+1
    print('laywer')
    print(level)
    return bottom,father,start_end,top,belong,level
def transfer(relationship,bom):
    bottom, father, start_end, top,belong,laywer=pre_data(relationship,bom)
    ans={}
    #识别底层活动
    for i in bottom:
        if relationship[i]!=[] and all(acti not in bom for acti in relationship[i]):#有紧前，紧前无子
            ans[i]=relationship[i]
        elif relationship[i]!=[]:#有紧前，紧前有子
            ans[i]=[]
            for pre in relationship[i]:
                if pre in bom:
                    end=start_end[pre]['end_activity']
                    while any(acti in bom for acti in end):
                        end_copy=end.copy()
                        for acti in end :
                            if acti in bom:
                                end_copy.remove(acti)
                                end_copy+=start_end[acti]['end_activity']
                        end=end_copy
                    ans[i]+=end
                else:
                    ans[i].append(pre)
        else:#无紧前
            ans[i]=[]
            fa=father[i]#找到父件（底层一定有父）
            while relationship[fa]==[] and fa not in top:
                fa=father[fa]
            #找到父件紧前
            for pre in relationship[fa]:
                if pre in bom:
                    end=start_end[pre]['end_activity']
                    while any(acti in bom for acti in end):
                        end_copy=end.copy()
                        for acti in end :
                            if acti in bom:
                                end_copy.remove(acti)
                                end_copy+=start_end[acti]['end_activity']
                        end=end_copy
                    ans[i]+=end
                else:
                    ans[i].append(pre)
            #检查父件的紧前
    return ans
def get_next_acti(relationship):
    next_acti={}
    for key,value in relationship.items():
        if len(value)!=0:
            for v in value:
                if v in next_acti:
                    next_acti[v].append(key)
                else:
                    next_acti[v]=[key]
    for key in relationship:
        if key not in next_acti:
            next_acti[key]=[]
    print('next_acti')
    print(next_acti)
    return next_acti


def get_all_successors(activity, successors_dict):
    """
    获取指定活动之后的所有后续活动集合。
    
    :param activity: 指定的活动
    :param successors_dict: 紧后关系字典
    :return: 一个集合，包含指定活动之后的所有后续活动
    """
    def dfs(current_activity):
        # 将当前活动添加到结果集合中
        result.add(current_activity)
        # 遍历当前活动的紧后活动
        for successor in successors_dict.get(current_activity, []):
            # 如果紧后活动还没有被访问过，递归地继续搜索
            if successor not in result:
                dfs(successor)
    
    result = set()  # 初始化结果集合
    dfs(activity)  # 从指定活动开始深度优先搜索
    return result



# Press the green button in the gutter to run the script
if __name__ == '__main__':
    bottom,father,start_end,top,belong,level=pre_data(relationship,bom)
    print('shuliang')
    print(len(start_end))
    print(len(level))
    bottom_relationship = transfer(relationship, bom)
    print('bottom_relstionship')
    print(bottom_relationship)
        
    next_acti=get_next_acti(bottom_relationship)
    res={}
    for key,value in point.items():
        all_successors = get_all_successors(point[key], next_acti)
        res[key]=all_successors
    print('合同生效后的活动')
    print(res)