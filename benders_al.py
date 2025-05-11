'''
Author: tu155 13293356554@163.com
Date: 2025-03-02 16:05:37
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-04 14:45:39
FilePath: \项目计划\benders_al.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from rsome import dro
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

data = pd.read_csv('https://xiongpengnus.github.io/rsome/taxi_rain.csv')
D,V=data.iloc[:,:10],data.iloc[:,10:]# D:demand &V:side information

regr =DecisionTreeRegressor(max_leaf_nodes=4,min_samples_leaf=3)
# max leaf nodes# min sample size of each leaf
regr.fit(V,D)
mu,index,counts=np.unique(regr.predict(V),axis=0,# conditional mean
                          return_inverse=True,
                          return_counts=True)
w= counts/V.shape[0]    # scenario weights
# conditional variance
phi = np.array([D.values[index==i].var(axis=0) for i in range(len(counts))])
# upperbound ofeach scenari
d_ub = np.array([D.values[index==i].max(axis=0) for i in range(len(counts))])
# lowerbound of each scenari
d_lb = np.array([D.values[index==i].min(axis=0) for i in range(len(counts))])

from rsome import square
from rsome import E#期望
from rsome import grb_solver as grb

S=len(counts)#情景个数

I, J = 1, 10#需求、供应节点数量
#需求节点单位需求被满足的收益
r = np.array([4.50, 4.41, 3.61, 4.49, 4.38, 4.58, 4.53, 4.64, 4.58, 4.32])
#成本系数
c = 3 * np.ones((I, J))
#汽车的最大供应量
q = 400 * np.ones(I)
model = dro.Model(S)#有S个情景的DRO模型
d =model.rvar(J)#随机需求变量d
u= model.rvar(J)#辅助随机变量u
fset = model.ambiguity()#创建一个模糊集
for s in range(S):
    fset[s].exptset(E(d)== mu[s],
                    E(u)<= phi[s])#指定随机决策变量和辅助变量的期望
    fset[s].suppset(d >= d_lb[s],
                    d <= d_ub[s],
                    square(d-mu[s])<= u)#指定d和u的支撑集合
pr = model.p# an array of scenario probabilities
fset.probset(pr == w) # w as scenario weights赋值

x= model.dvar((I,J))
y = model.dvar(J)
y.adapt(d)#
y.adapt(u)
for s in range(S):
    y.adapt(s)

model.minsup(((c-r)*x).sum()+E(r@y),fset)
model.st(y>=x.sum(axis=0)-d,y>=0)
model.st(x>=0,x.sum(axis=1)<=q)

model.solve(solver=grb,display=True, log=False, params={})
# solve the model by Gurobi
objval = model.get()
# get the optimal objective value
xsol = x.get()
status = model.solution.status
stime =model.solution.time

optimal_y_value=y.get()
print(f'The optimal objective value:{objval}')
print('optimal_x_value:')
print(xsol)
print('optimal_y_value:')#y是和情景相关的
print(optimal_y_value)
# get the optimal solution
# return the solution status
# return the solution time