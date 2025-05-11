from datetime import datetime 
import pandas as pd


def read_excel_project():
    path=r'input/aps_project.xlsx'
    df = pd.read_excel(path)

    #初始化数据
    due_time={}#项目交期
    project_status={}#项目状态
    
    row=['project_id','start_time','end_time','status']
    for _,row in df.iterrows():
        project_id=str(row['project_id'])
        # start_time=str(row['start_time'])
        end_time=row['end_time'].date()
        status=str(row['status'])
        if project_id not in due_time:
            due_time[project_id]=end_time
        if status=='NEW':
            project_status[project_id]=False
        else:   
            project_status[project_id]=True
    print('due_time,project_status')
    print(due_time,project_status)
    return due_time,project_status

def read_excel_activity():
    path=r'input/aps_activity.xlsx'
    df = pd.read_excel(path)
    # row=['project_id','activity_id','start_time','end_time','parent_activity_id','proceeding_activity_id','durance','resource_id','attribute1','activity_name']

    point={}
    project_acti={}
    relationship_dict={}
    father={}
    durance={}
    resource={}
    have_worked={}
    type_dict={}
    for _,row in df.iterrows():
        # print('========')
        # print(row[2])
        # print(type(row[2]))#<class 'datetime.date'>
        project=str(row['project_id'])
        activity=str(row['activity_id'])
        par_acti=str(int(row['parent_activity_id'])) if str(row['parent_activity_id'])!='nan' else 'nan'
        proc_acti=str(int(row['proceeding_activity_id'])) if str(row['proceeding_activity_id'])!='nan' else 'nan'
        dur=int(row['durance']) if int(row['durance']) >0 else 5
        rec=str(row['resource_id'])
        status=str(row['attribute1'])
        sigle_type=str(row['activity_type'])
        if project in project_status:
            if project not in project_acti:
                project_acti[project]=[activity]
                relationship_dict[project]=[]#补充项目在关系表里
            else:
                project_acti[project].append(activity)
            if par_acti=='nan':
                father[activity]=project
            else:
                father[activity]=par_acti

            if proc_acti=='nan':
                relationship_dict[activity]=[]
            else:
                relationship_dict[activity]=[proc_acti]
            
            durance[activity]=dur
            resource[activity]=rec
            # resource[activity]={rec:1}

            #已经完成加工的活动
            if status=='COMPLERED':
                have_worked[activity]={}
                have_worked[activity]['start_time']=row[2]
                have_worked[activity]['end_time']=row[3]
            if row['activity_name']=='合同生效':
                point[project]=activity

            type_dict[activity]=sigle_type

    bom={}
    for son,fath in father.items():
        if fath not in bom:
            bom[fath]=[son]
        else:
            bom[fath].append(son)

    # print('project_acti,relationship_dict,father,durance,resource,have_worked,point,bom')
    # print(project_acti,relationship_dict,father,durance,resource,have_worked,point,bom)
    print('project_acti',project_acti,'\n\n',
      'relationship_dict',relationship_dict,'\n\n',
      'father',father,'\n\n',
      'durance',durance,'\n\n',
      'resource',resource,'\n\n',
      'have_worked',have_worked,'\n\n',
      'point',point,'\n\n',
      'bom',bom)
    return project_acti,relationship_dict,father,durance,resource,have_worked,point,bom,type_dict# 返回包含数据的字典

def read_excel_resource():
    path=r'input/bl_cx_prod_line.xlsx'
    df = pd.read_excel(path)
    resource_capacity={}
    for _,row in df.iterrows():
        prod_line_id=str(row['prod_line_id'])
        if prod_line_id not in resource_capacity:
            resource_capacity[prod_line_id]=int(row['attribute5'])
        else:
            resource_capacity[prod_line_id]=20

    return resource_capacity
resource_capacity=read_excel_resource()
def read_excel_calendar():
    # 读取Excel文件，由于数据库当前没有生产线信息，所以使用excel文件
    path=r'input/mes_cal_teamshift.xlsx'
    df = pd.read_excel(path)
    
    # 初始化字典
    resource_calendar = {}
    
    # 遍历DataFrame的每一行
    for _, row in df.iterrows():
        # 获取日期并转换为datetime.date对象
        the_day = pd.to_datetime(row['the_day']).date()
        prod_line_id = str(row['prod_line_id'])
        work_flag = True if row['work_flag'] == 'Y' else False
        
        # 如果prod_line_id不在字典中，创建一个新的字典
        if prod_line_id not in resource_calendar:
            resource_calendar[prod_line_id] = {}
        
        # 添加日期和对应的work_flag
        resource_calendar[prod_line_id][the_day] = work_flag
    
    # 找出拥有最多日期工作标志的生产线
    max_dates_line = max(resource_calendar.items(), key=lambda x: len(x[1]))[0]
    max_dates = resource_calendar[max_dates_line]
    
    # 补全其他生产线的日期
    for prod_line_id in resource_calendar:
        if prod_line_id != max_dates_line:
            for date in max_dates:
                if date not in resource_calendar[prod_line_id]:
                    resource_calendar[prod_line_id][date] = max_dates[date]
    #补充calendar缺失数据
    for i in resource_capacity:
        if i not in resource_calendar:
            resource_calendar[i] = resource_calendar[max_dates_line]
    print("resource_calendar")
    print(resource_calendar)
    return resource_calendar

due_time,project_status= read_excel_project()

project_acti,relationship_dict,father,durance,resource,have_worked,point,bom,type_dict=read_excel_activity()
print(project_acti.keys())

resource_calendar = read_excel_calendar()

print('数据读取已完成')