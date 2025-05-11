import pymssql  # 导入 pymssql 库，用于连接 SQL Server 数据库
from datetime import datetime 
import pandas as pd
def sql_project():  # 定义一个函数 sql_connect，用于数据库连接和数据获取
    # 使用 pymssql.connect 方法建立数据库连接
    conn = pymssql.connect(
        host='192.168.197.193',  # 数据库服务器的 IP 地址
        port='1433',  # SQL Server 的端口号，默认为 1433
        user='wcs',  # 数据库用户名
        password='wcs`193',  # 数据库密码
        database='mes',  # 要连接的数据库名
        charset='GBK'  # 指定字符集，这里使用 GBK，适用于中文编码
    )
    cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 命令

    # 定义查询的列
    # 将 qty 列转换为整数类型，并将 consume_date 列转换为格式化的字符串
    # columns = 'project_id, CAST(qty AS int) AS qty_int, CONVERT(varchar, consume_date, 23) AS consume_date_formatted'
    columns = 'project_id, start_time, end_time, status'
    
    # 定义 SQL 查询语句，从指定的表中查询数据
    sql = f'SELECT {columns} FROM [dbo].[aps_project]'
    cursor.execute(sql)  # 执行 SQL 查询
    results = cursor.fetchall()  # 获取查询结果
    
    #初始化数据
    due_time={}#项目交期
    project_status={}#项目状态
    
    #按行处理查询结果
    for row in results:
        key=str(row[0])#整型转化为字符串
        status=row[3]
        due_time[key]=row[2]#<class 'datetime.date'>
        if status=='NEW':#项目合同未生效，只对合同生效前的活动做计划
            project_status[key]=False
        else:#项目合同已生效，对项目所有活动做计划
            project_status[key]=True
    print('due_time,project_status')
    print(due_time,project_status)
    
    # 关闭游标和数据库连接，释放资源
    cursor.close()
    conn.close()
    
    return due_time,project_status

def sql_activity():
    # 使用 pymssql.connect 方法建立数据库连接
    conn = pymssql.connect(
        host='192.168.197.193',  # 数据库服务器的 IP 地址
        port='1433',  # SQL Server 的端口号，默认为 1433
        user='wcs',  # 数据库用户名
        password='wcs`193',  # 数据库密码
        database='mes',  # 要连接的数据库名
        charset='GBK'  # 指定字符集，这里使用 GBK，适用于中文编码
    )
    cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 命令

    activity_columns='project_id,activity_id,start_time,end_time,parent_activity_id,proceeding_activity_id,durance,resource_id,attribute1,activity_name'
    # activity_columns='project_id,activity_id'
    
    sql2=f'SELECT {activity_columns} FROM [dbo].[aps_activity]'
    cursor.execute(sql2)
    activity_results=cursor.fetchall()
    activity_status=['NEW','SCHEDULE','RELEASED','WORKING','COMPLETED']
    
    point={}
    project_acti={}
    relationship_dict={}
    father={}
    durance={}
    resource={}
    have_worked={}
    for row in activity_results:
        # print('========')
        # print(row[2])
        # print(type(row[2]))#<class 'datetime.date'>
        project=str(row[0])
        activity=str(row[1])
        par_acti=str(row[4])
        proc_acti=str(row[5])
        dur=int(row[6]) if int(row[6]) >0 else 5
        rec=str(row[7])
        status=str(row[8])
        if project in project_status:
            if project not in project_acti:
                project_acti[project]=[activity]
                relationship_dict[project]=[]#补充项目在关系表里
            else:
                project_acti[project].append(activity)
            if par_acti=='None':
                father[activity]=project
            else:
                father[activity]=par_acti

            if proc_acti=='None':
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
            if row[9]=='合同生效':
                point[project]=activity
        
    bom={}
    for son,fath in father.items():
        if fath not in bom:
            bom[fath]=[son]
        else:
            bom[fath].append(son)
    # print(resource)
    # 关闭游标和数据库连接，释放资源
    cursor.close()
    conn.close()
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
    return project_acti,relationship_dict,father,durance,resource,have_worked,point,bom# 返回包含数据的字典
def sql_resource():
    # 使用 pymssql.connect 方法建立数据库连接
    conn = pymssql.connect(
        host='192.168.197.193',  # 数据库服务器的 IP 地址
        port='1433',  # SQL Server 的端口号，默认为 1433
        user='wcs',  # 数据库用户名
        password='wcs`193',  # 数据库密码
        database='mes',  # 要连接的数据库名
        charset='GBK'  # 指定字符集，这里使用 GBK，适用于中文编码
    )
    cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 命令
    columns='prod_line_id,attribute5'
 
    sql=f'SELECT {columns} FROM [dbo].[bl_cx_prod_line]'
    cursor.execute(sql)
    results=cursor.fetchall()
    resource_capacity={}
    for row in results:
        prod_line_id=str(row[0])
        if row[1] is not None:
            resource_capacity[prod_line_id]=int(row[1])
        else:
            resource_capacity[prod_line_id]=20

    # 关闭游标和数据库连接，释放资源
    cursor.close()
    conn.close()
    print("resource_capacity")
    print(resource_capacity)
    return resource_capacity
def sql_cal():
    # 使用 pymssql.connect 方法建立数据库连接
    conn = pymssql.connect(
        host='192.168.197.193',  # 数据库服务器的 IP 地址
        port='1433',  # SQL Server 的端口号，默认为 1433
        user='wcs',  # 数据库用户名
        password='wcs`193',  # 数据库密码
        database='mes',  # 要连接的数据库名
        charset='GBK'  # 指定字符集，这里使用 GBK，适用于中文编码
    )
    cursor = conn.cursor()  # 创建一个游标对象，用于执行 SQL 命令
    columns='the_day,prod_line_id,work_flag'
    
    sql=f'SELECT {columns} FROM [dbo].[mes_cal_teamshift]'
    cursor.execute(sql)
    results=cursor.fetchall()
    resource_calendar={}
    for row in results: 
        the_day=datetime.strptime(row[0],'%Y-%m-%d').date()
        prod_line_id=str(row[1])
        work_flag=True if str(row[2])=='Y' else False
        if prod_line_id not in resource_calendar:
            resource_calendar[prod_line_id]={}
            resource_calendar[prod_line_id][the_day]=work_flag
        else:
            resource_calendar[prod_line_id][the_day]=work_flag
    

    # 关闭游标和数据库连接，释放资源
    cursor.close()
    conn.close()
    print("resource_calendar")
    print(resource_calendar)
    return resource_calendar

# if __name__=="__main__":
    # 调用函数并获取结果
due_time,project_status= sql_project()
# print(due_time,'\n\n',project_status)  # 打印结果
# print('-----------')
project_acti,relationship_dict,father,durance,resource,have_worked,point,bom=sql_activity()
print(project_acti.keys())
# print('----------')
resource_capacity=sql_resource()
# print(resource_capacity)
'''
resource_calendar=sql_cal()

'''

def read_excel_calendar():
    # 读取Excel文件，由于数据库当前没有生产线信息，所以使用excel文件
    df = pd.read_excel('mes_cal_teamshift.xlsx')
    
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

resource_calendar = read_excel_calendar()

print('数据读取已完成')