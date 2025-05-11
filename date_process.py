from datetime import date, timedelta

def date_process(now: date, plus: int, calendar: dict):
    new_date = now
    days_added = 0  # 记录实际增加的天数
    while days_added < plus:  # 循环直到加上所需的天数
        new_date += timedelta(days=1)
        if new_date in calendar and calendar[new_date]:  # 如果新日期是工作日
            days_added += 1  # 增加实际天数
    return new_date
def back_date_process(end:date,minus:int,calendar:dict):
    new_date=end
    days_minus=0
    while days_minus<minus:
        new_date-=timedelta(days=1)
        if new_date in calendar and calendar[new_date]:
            days_minus+=1
    return new_date
# 设置初始日期
now = date(2024, 12, 30)
# 需要增加的天数
plus = 5
# 工作日历，False表示非工作日
calendar = {
    date(2024, 12, 30): True,
    date(2024, 12, 31): False,
    date(2025, 1, 1): False,
    date(2025, 1, 2): True,
    date(2025, 1, 3): True,
    date(2025, 1, 4): True,
    date(2025, 1, 5): True,
    date(2025, 1, 6): True,
    date(2025, 1, 7): True,
}
def date_match(now_date,lenth,calendar):
    num_calendar={}
    accumulated_no_work_days={}
    num=0
    for i in range(int(lenth)):
        num_calendar[i]=calendar[now_date+timedelta(days=i)]
        if not calendar[now_date+timedelta(days=i)]:
            num+=1
        accumulated_no_work_days[i]=num
    return num_calendar,accumulated_no_work_days
def date_to_int(now_date,date):
    lenth=(date-now_date).days
    return lenth
def int_to_date(now_date,num):
    return now_date+timedelta(days=num)

if __name__=="__main__":
    # 计算新的日期
    new_date = date_process(now, plus, calendar)
    end=date(2025,1,2)
    start=back_date_process(end,1,calendar)
    print(start)
    num_calendar,accumulated_no_work_days=date_match(now,new_date,calendar)
    a=date_to_int(date(2025, 1, 7),date(2025, 1, 9))
    print(a)
    b=int_to_date(date(2025, 1, 7),3)
    print(b)