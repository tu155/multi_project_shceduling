'''
Author: tu155 13293356554@163.com
Date: 2025-03-03 10:51:42
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-10 19:52:08
FilePath: \项目计划\easy_rsome.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from rsome import ro                # import the ro modeling tool
from rsome import grb_solver as grb
model = ro.Model('LP model')        # create a Model object
x = model.dvar()                    # define a decision variable x
y = model.dvar()                    # define a decision variable y

model.max(3*x + 4*y)                # maximize the objective function
model.st(2.5*x + y <= 20)           # specify the 1st constraints
model.st(5*x + 3*y <= 30)           # specify the 2nd constraints
model.st(x + 2*y <= 16)             # specify the 3rd constraints
model.st(abs(y) <= 2)               # specify the 4th constraints

model.solve(grb)                    # solve the model by the default solver
optimal_objective_value=model.get()
The_optimal_solution_of_the_variable =[x.get(),y.get()]
print(f'The optimal objective value:{optimal_objective_value}')
print(f'The optimal solution of the variable:{The_optimal_solution_of_the_variable}')
