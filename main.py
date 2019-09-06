#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 20:24:12 2019

@author: jack
"""

'''
该Python程序用以测试所有群体智能算法
每个算法运行 ? 次, 并完成可视化
'''
import numpy as np
from models import SA_TSP
from models import GA_TSP
from models import PSO_TSP

SA_run = SA_TSP(data_name='gr202.tsp', max_iter=10000, T_max=3000000, T_min=1e-20)
GA_run = GA_TSP(data_name='gr202.tsp', max_iter=50)
PSO_run = PSO_TSP(data_name='gr202.tsp', max_iter=50)


def run_model(some_run, times=1):
    # 此函数运行所有模型
    # 返回 含有100个元素的列表
    model_list = []
    for i in range(times):
        some_run.main()
        print(some_run.T_K)
        model_list.append( str(some_run.get_result()[1]) )
    return model_list

def list2str(output_list):
    # 该类将输出变为 latex 表格样式的字符串
    output_str = ''
    for i in range(10):
        output_str += '&'.join(output_list[(0+i*8):(8+i*8)])
        output_str += '\n'
        print(output_str)
        
# list2str(run_model(SA_run))
        
SA_run.main()
GA_run.main()
PSO_run.main()

SA_run.visualization_latlun()
GA_run.visualization_latlun()
PSO_run.visualization_latlun()
