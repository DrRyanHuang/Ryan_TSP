#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 03:01:40 2019

@author: Ryan
"""

'''
参加2019年数学建模
在失败 3 次之后这是我最后的机会, 我要去拼一把, 此处放下我准备的轮子

先暂存在这里, 有时间慢慢修补该项目

在面向对象的过程中, 写函数的时候我尽量解耦合
'''

import numpy as np
from collections.abc import Iterable


def subStrCount(str1, sub):
    '''
    在子字符串首尾没有重叠时, 与str.count相同
    在有重叠时, 可以尝试用此函数
    鄙人才疏学浅, 没找到对应的API, 若有请告诉我, 多谢
    str1 是 原字符串
    sub 是 子字符串
    '''
    counter = 0  # 计数器
    index = 0
    sub_length = len(sub)
    while(True):
        index = str1.find(sub, index)
        if index==-1: # 若没有找到则返回 -1
            break        
        if str1[index:index+sub_length] == sub:
            counter += 1
        index += 1        
    return counter


class GM_1_1:
    def __init__(self, array):
        self.array = array
        self.data_check(self.array)
        
    def calc_param(self, array):
        '''
        array 一维数组: 可以传入一维列表或者一维numpy数组
        
        GM(1,1)表示模型是一阶微分方程, 且只有一个变量的灰色模型
        
        参考《数学建模算法与应用》——司守奎、孙兆亮 P399
        '''
        
        # 先有效性检验
        assert isinstance(array, Iterable), "传入的array参数不可迭代"
        
        x_0 = array.copy()
        
        # 一次累加生产序列（1-AGO）
        x_1 = [np.sum(x_0[:(i+1)]) for i,_ in enumerate(x_0)]
        x_1 = np.array(x_1)
        
        # 1-AGO 的均值生成序列
        z_1 = [(x_1[i]+x_1[i+1])/2 for i in range(len(x_1)-1)]
        z_1 = np.array(z_1)
        
        '''
        建立灰微分方程
            x_0(k) + a*z_1(k) = b   , k=2,3,...,n
        相应的白化微分方程为
            d(x_1) / dt + a*x_1(t) = b
        而我们现在就是要求解 参数 a和b的估计值
        '''
        
        # 列写矩阵 Y, B
        Y_matrix = x_0[1:].reshape(1,-1).T
        
        B_matrix = np.concatenate([-z_1.reshape(-1,1), 
                                   np.ones(shape=(len(z_1), 1))],  # 注意 z_1 是从2开始的
                                  axis=1)
        # 求得u估计, 得到a, b
        u_head = np.matmul(np.matmul(np.linalg.inv(np.matmul(B_matrix.T, B_matrix)), B_matrix.T), Y_matrix)  # numpy 没有多矩阵相乘的API吗, 只能递归相乘？？
    
        a_head, b_head = u_head.reshape(-1)
        return a_head, b_head
        
    def data_check(self, array):
        '''
        array 一维数组: 可以传入一维列表或者一维numpy数组
        
        为了保证建模方法的可靠性, 需要对已知数据列做必要的检验处理
        '''
        
        n = len(array)
        # 求解级比
        lambda_k = [array[i]/array[i-1] for i in range(1, n)]
        # print(lambda_k)
        # print(np.exp(-2/(n+1)))
        # print(np.exp(2/(n+2)))
        assert (lambda_k>np.exp(-2/(n+1))).all(), "下限不满足, 请调参"
        assert (lambda_k<np.exp(2/(n+2))).all(), "上限不满足, 请调参"
    

    
class GM_2_1:
    def __init__(self):
        pass
    
    def calc_param(self, array):
        '''
        array 一维数组: 可以传入一维列表或者一维numpy数组
        
        GM(2,1)模型适用于非单调的摆动发展序列或有饱和的S型序列
        
        参考《数学建模算法与应用》——司守奎、孙兆亮 P403
        '''
        
        # 先有效性检验
        assert isinstance(array, Iterable), "传入的array参数不可迭代"
        
        x_0 = array.copy()
        
        # 一次累加生产序列（1-AGO）
        x_1 = [np.sum(x_0[:(i+1)]) for i,_ in enumerate(x_0)]
        x_1 = np.array(x_1)
        # print(x_1)
        
        # 1-AGO 的均值生成序列
        z_1 = [(x_1[i]+x_1[i+1])/2 for i in range(len(x_1)-1)]
        z_1 = np.array(z_1)
        # print(z_1)
        
        # x_0 的一次累减生成序列 ( 1-IAGO )alpha_1_x_0
        alpha_1_x_0 = [(x_0[i+1]-x_0[i]) for i in  range(len(x_0)-1)]
        alpha_1_x_0 = np.array(alpha_1_x_0)
        # print(alpha_1_x_0)
        
        '''
        建立灰微分方程
            alpha_1_x_0(k) + a_1*x_0 + a_2*x_1 = b
        白化方程
            d2(x_1)/dt_2 + a_1*dx_1/dt + a_2*x_1 = b
        '''
        B_matrix = np.concatenate([-x_0[1:].reshape(-1,1),
                                   -z_1.reshape(-1,1), 
                                   np.ones(shape=(len(z_1), 1))],  # 注意 z_1 是从2开始的
                                  axis=1)
        # print(B_matrix)
        Y_matrix = alpha_1_x_0.reshape(-1, 1)
        # print(Y_matrix)
        
        # 求得u估计, 得到a_1, a_2, b
        u_head = np.matmul(np.matmul(np.linalg.inv(np.matmul(B_matrix.T, B_matrix)), B_matrix.T), Y_matrix) 
        # print(u_head)
        
        a_1, a_2, b = u_head.reshape(-1)
        return a_1, a_2, b
    
class Markov: # 马尔科夫预测
    def getStateSpace(self, sequence):
        '''
        得到状态空间
        传入的序列为列表、字符串、np.array等
        打印状态空间
        
        '''
        assert isinstance(sequence, (list, np.ndarray, str)), '序列应为列表、字符串、np.ndarray等'
        return list(set(sequence))
    
    def getStateConvertMatrix(self, sequence, element_list):
        '''
        将一切序列转换成 str, 通过 str.count 函数来计算
        sequence 原始序列
        element_list 状态空间
        返回 一步转移概率矩阵 stateConvertMatrix
        '''
        state_n = len(element_list)                # 状态个数
        sequence_str = str(sequence)               # 将所有的状态序列转移为 str 
        stateConvertMatrix = np.zeros(shape=[state_n]*2)    # 初始化状态转移矩阵

        
        for i in range(len(element_list)):
            for j in range(len(element_list)):
                stateConvertMatrix[i][j] = subStrCount(sequence_str, element_list[i]+element_list[j])
            stateConvertMatrix[i] /= np.sum(stateConvertMatrix[i]) # 此处是原地处理
        return stateConvertMatrix
    
a = Markov.getStateConvertMatrix('','00010', ['1', '0'])


