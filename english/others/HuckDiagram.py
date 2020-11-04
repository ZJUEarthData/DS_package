'''  LOG:
    coder:Yifan Zhang
    built time 2020-08-07
    !/HuckDiagram.py
    --*—— coding: utf-8 -*-  '''

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# %matplotlib inline

# 使matplotlib Display Chinese characters on the diagram
from matplotlib import font_manager

# Define global variables
my_font = font_manager.FontProperties(fname = "C:\Windows\Fonts\msyh.ttc")
q = pd.read_excel(".\\High quality 2486 main.xlsx",sheet_name = 1)

#Define the class
class HuckDiagram:


    # Passing parameters
    def __init__(self, my_font, q):
        self.my_font = my_font
        self.q = q

    # Define a function
    def plot(self):

        #Call parameters
        my_font = self.my_font
        q = self.q

        #q1,q2
        q.fillna(0, inplace = True)
        q.info()
        q1 = q[q["TRUE VALUE"] == 222]
        q2 = q[q["TRUE VALUE"] == 111]

        #Relationship between the main element (Y-axis) and silica (X-axis)
        name = ["TIO2(WT%)", "AL2O3(WT%)", "CR2O3(WT%)", "FEOT(WT%)",
                "CAO(WT%)", "MGO(WT%)", "MNO(WT%)", "K2O(WT%)", "NA2O(WT%)"]
        j = 1
        fig = plt.figure(figsize=(16, 16))
        for i in name:
            fig.add_subplot(3, 3, j)
            j = j + 1
            plt.plot(q1["SIO2(WT%)"], q1[i], ".", label="not metasomatism")
            plt.plot(q2["SIO2(WT%)"], q2[i], ".", label="metasomatism")
            plt.xlabel("SIO2(WT%)")
            plt.ylabel(i)
            plt.legend(prop = my_font)
            #Save the scatterplot to the specified path file((PS: Add custom path name at ellipsis)）
            plt.savefig(r'....\bulk.png', dpi=300)
            #visualization
            plt.show()

        #Relationship between some trace element index (Y) and silica (X)
        fig = plt.figure(figsize = (16, 6))
        fig.add_subplot(1, 2, 1)
        plt.plot(q1["SIO2(WT%)"], q1["Ca/Al"], ".", label = "not metasomatism")
        plt.plot(q2["SIO2(WT%)"], q2["Ca/Al"], ".", label = "metasomatism")
        plt.xlabel("SIO2(WT%)")
        plt.ylabel("Ca/Al")
        plt.legend(prop = my_font)
        fig.add_subplot(1, 2, 2)
        plt.plot(q1["SIO2(WT%)"], q1["#mg"], ".", label = "not metasomatism")
        plt.plot(q2["SIO2(WT%)"], q2["#mg"], ".", label = "metasomatism")
        plt.xlabel("SIO2(WT%)")
        plt.ylabel("#mg")
        plt.legend(prop = my_font)
        # Saves the scatter diagram to the specified path file
        plt.savefig(r'....\microelement.png', dpi=300)
        # visualization
        plt.show()

        #Select characteristic data for #mg to draw diagram
        name2 = ["Ca/Al", "K2O(WT%)", "NA2O(WT%)"]
        n = 1
        fig = plt.figure(figsize = (8, 16))
        for m in name2:
            fig.add_subplot(3, 1, n)
            n = n + 1
            plt.plot(q1["#mg"], q1[m], ".", label = "not metasomatism")
            plt.plot(q2["#mg"], q2[m], ".", label = "metasomatism")
            plt.xlabel("#mg")
            plt.ylabel(m)
            plt.legend(prop = my_font)
            # Saves the scatter diagram to the specified path file
            plt.savefig(r'....\characteristic.png', dpi=300)
            # visualization
            plt.show()

# Call the class function plot() output
diagram = HuckDiagram(my_font, q)
diagram.plot()
print(diagram)
