'''
Author: tu155 13293356554@163.com
Date: 2025-03-18 11:13:36
LastEditors: tu155 13293356554@163.com
LastEditTime: 2025-03-18 16:11:42
FilePath: \项目计划\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from data_manager import RCPSPData
from main_problem import MainProblem
from datetime import datetime
def main():
    # 初始化数据
    data = RCPSPData()
    
    # 验证数据
    if not data.validate_data():
        print("\n数据验证失败，请检查上述错误信息")
        return
    
    print("\n=== 开始求解主问题 ===")
    # 创建并求解主问题
    main_prob = MainProblem(data)
    main_prob.build_model()
    
    # 设置CVaR约束
    cvar_loss = 1002
    main_prob.add_cvar_constraint(cvar_loss)
    
    # 求解并获取结果
    results = main_prob.solve()
    
    if results is None:
        print("\n模型求解失败，请检查model.ilp文件中的不可行约束")
        return
        
    print(f"\nCVaR_loss: {results['objective_value']:g}")
    print(f"\n结果已保存到 'optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx' 文件中")

if __name__ == '__main__':
    main() 