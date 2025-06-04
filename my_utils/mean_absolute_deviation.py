#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


avg_ind = [0.8426, 0.7701, 0.8079, 0.7828, 0.8394, 0.7868, 0.7844, 0.8108, 0.7913, 0.8137]
max_seq =  [0.8582, 0.7735, 0.8127, 0.7828, 0.8515, 0.8166, 0.8065, 0.8176, 0.788, 0.8044]
max_ind = [0.796, 0.727, 0.706, 0.775, 0.784, 0.84, 0.761, 0.817, 0.733, 0.758]
rand_ind =  [0.8482, 0.7651, 0.8063, 0.7828, 0.829, 0.8072, 0.7793, 0.8193, 0.788, 0.8247]
# equi =  [0.825, 0.804, 0.842, 0.8405, 0.8385]


def mean_absolute_deviation(data):
    """
    计算平均绝对偏差 (Mean Absolute Deviation)
    
    参数:
        data: 数值列表或NumPy数组
        
    返回:
        平均绝对偏差值
    """
    # 转换输入数据为NumPy数组
    data = np.array(data)
    
    # 计算平均值
    mean = np.mean(data)
    
    # 计算每个数据点与平均值的绝对偏差
    absolute_deviations = np.abs(data - mean)
    
    # 计算平均绝对偏差
    mad = np.mean(absolute_deviations)
    
    return mad

if __name__ == "__main__":
    # 计算并打印预定义列表的MAD值
    print("预定义列表的平均绝对偏差 (MAD):")
    print(f"avg_ind: {mean_absolute_deviation(avg_ind):.6f}")
    print(f"max_seq: {mean_absolute_deviation(max_seq):.6f}")
    print(f"max_ind: {mean_absolute_deviation(max_ind):.6f}")
    print(f"rand_ind: {mean_absolute_deviation(rand_ind):.6f}")
    print(f"equi: {mean_absolute_deviation(equi):.6f}")
    
    # 以下是原有的用户输入功能
    try:
        # 从用户获取输入
        print("\n是否要计算其他数值的MAD？(y/n):")
        choice = input().strip().lower()
        
        if choice == 'y':
            print("请输入一组数值，用空格分隔:")
            user_input = input()
            
            # 将输入转换为浮点数列表
            data = [float(x) for x in user_input.split()]
            
            if len(data) < 2:
                print("错误: 请至少输入两个数值")
            else:
                # 计算并显示结果
                result = mean_absolute_deviation(data)
                print(f"平均绝对偏差 (MAD): {result:.6f}")
                
    except ValueError:
        print("错误: 请确保输入的是有效数值")
    except Exception as e:
        print(f"发生错误: {e}") 