#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 20:12:43 2019

@author: jack
"""

import numpy as np
from models import some_TSP
import random
import warnings
class ant:
    def __init__(self, city_n):
        '''
        蚁群算法中的小蚂蚁类, 初次写蚁群算法, 请多海涵
        city_n  : 城市数量

        '''
        self.city_n = city_n
        self.start = np.random.randint(low=0, high=city_n) # 前包后不包
        self.city_record_list = [self.start]
        
    def choose_next_city(self, local_city, P_array):
        '''
        暂且以轮盘赌博的形式选择路径
        尽管以我的经验, 十有八九不收敛
        '''
        P_array_copy = np.concatenate((P_array[:local_city], P_array[local_city:]))  # 若 P_array 不是一维的应该就废了, 不能用 np.concatenate
        choose_result = random.choices(population = P_array_copy, 
                                     weights = P_array_copy/np.sum(P_array_copy)) # 不得不说这个 random.choices 还是省了不少事情的
        return choose_result
        
class ACO_TSP(some_TSP):
    def __init__(self, data_name='gr202.tsp', alpha=1, beta=5, rho=0.1, Q=100, ant_n=None):
        # print(self.__class__.__name__) # 打印类的名字
        super().__init__(data_name)
        self.alpha = alpha                      # 信息素启发因子，表示在搜索中，信息量的相对重要程度
        self.beta = beta                        # 期望启发因子，表示在搜索中，启发式信息的相对重要程度
        self.rho = rho                          # 信息素挥发因子，表示信息素的持久度
        self.Q = Q                              # 常亮, 表示信息素浓度
        self.ant_n = 100 if None else ant_n     # 蚂蚁的数量
        self.pheromones = np.ones(shape=(self.city_n, )*2)  # 初始化信息素浓度矩阵
        # self.relation_distance_array
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 此处有 RuntimeWarning(除以0错误) 是暂时忽略警告
            self.probability = np.power(self.pheromones, self.alpha) * np.power(self.relation_distance_array, -self.beta)
    
    def update(self):
        '''
        更新信息素浓度
        '''
        pass
    
    def main(self, ):
        #for 
        pass
    
print(ACO_TSP().probability[0])