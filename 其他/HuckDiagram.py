'''  LOG:
    coder:张一凡
    built time 2020-08-07
    !/HuckDiagram.py
    --*—— coding: utf-8 -*-  '''

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

# 使matplotlib 在图上显示中文字符
from matplotlib import font_manager

# 定义全局变量
my_font = font_manager.FontProperties(fname = "C:\Windows\Fonts\msyh.ttc")
q = pd.read_excel(".\\高质量2486条主量.xlsx",sheet_name = 1)

#定义类
class HuckDiagram:


    # 传参
    def __init__(self, my_font, q):
        self.my_font = my_font
        self.q = q

    # 定义函数
    def plot(self):

        #调用参数
        my_font = self.my_font
        q = self.q

        #q1,q2
        q.fillna(0, inplace = True)
        q.info()
        q1 = q[q["TRUE VALUE"] == 222]
        q2 = q[q["TRUE VALUE"] == 111]

        #主量元素(Y轴)与二氧化硅(X轴)的关系
        name = ["TIO2(WT%)", "AL2O3(WT%)", "CR2O3(WT%)", "FEOT(WT%)",
                "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)"]
        j = 1
        fig = plt.figure(figsize=(16, 16))
        for i in name:
            fig.add_subplot(3, 3, j)
            j = j + 1
            plt.plot(q1["SIO2(WT%)"], q1[i], ".", label="未交代")
            plt.plot(q2["SIO2(WT%)"], q2[i], ".", label="已交代")
            plt.xlabel("SIO2(WT%)")
            plt.ylabel(i)
            plt.legend(prop = my_font)
            #将散点图存入指定路径文件(ps:省略号处需补充自定义的路径名）
            plt.savefig(r'....\bulk.png', dpi=300)
            #可视化
            plt.show()

        #部分微量元素指标(Y)与二氧化硅(X)的关系
        fig = plt.figure(figsize = (16, 6))
        fig.add_subplot(1, 2, 1)
        plt.plot(q1["SIO2(WT%)"], q1["Ca/Al"], ".", label = "未交代")
        plt.plot(q2["SIO2(WT%)"], q2["Ca/Al"], ".", label = "已交代")
        plt.xlabel("SIO2(WT%)")
        plt.ylabel("Ca/Al")
        plt.legend(prop = my_font)
        fig.add_subplot(1, 2, 2)
        plt.plot(q1["SIO2(WT%)"], q1["#mg"], ".", label = "未交代")
        plt.plot(q2["SIO2(WT%)"], q2["#mg"], ".", label = "已交代")
        plt.xlabel("SIO2(WT%)")
        plt.ylabel("#mg")
        plt.legend(prop = my_font)
        # 将散点图存入指定路径文件
        plt.savefig(r'....\microelement.png', dpi=300)
        # 可视化
        plt.show()

        #选择有特征数据对#mg投图
        name2 = ["Ca/Al", "K2O(WT%)", "NA2O(WT%)"]
        n = 1
        fig = plt.figure(figsize = (8, 16))
        for m in name2:
            fig.add_subplot(3, 1, n)
            n = n + 1
            plt.plot(q1["#mg"], q1[m], ".", label = "未交代")
            plt.plot(q2["#mg"], q2[m], ".", label = "已交代")
            plt.xlabel("#mg")
            plt.ylabel(m)
            plt.legend(prop = my_font)
            # 将散点图存入指定路径文件
            plt.savefig(r'....\characteristic.png', dpi=300)
            # 可视化
            plt.show()

# 调用类函数plot()输出
diagram = HuckDiagram(my_font, q)
diagram.plot()
print(diagram)
