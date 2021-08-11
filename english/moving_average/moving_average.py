#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 2021/8/2
# @Author  : 何灿 Sany
# @File    : moving_average.py
# @Editor  : vim

import pandas as pd
import numpy as np
import os

# the path to store calculation results
WORKING_PATH = os.getcwd()
RESULT_PATH = os.path.join(WORKING_PATH, "results")
os.makedirs(RESULT_PATH, exist_ok=True)

def single_x_vs_y(x_data, y_data, step_size):
    """algorithm for moving mean and moving standard deviation
    :param x_data: data as X axis
    :param y_data: data as Y axis
    :param step_size: how wide every interval has
    :return moving_mean, moving_std, x_mid:
    """
    x_min, x_max = x_data.min(), x_data.max()
    # calculate how many segments a specific column whose range (x_max - x_min) divided by step_size has 
    intervals =  (x_max - x_min) / step_size 
    segments = intervals if intervals == int(intervals) else int(intervals) + 1
    print("The amount of segments divided by step_size:", segments)
    
    moving_mean = []
    moving_std = []
    x_mid = []
    accum_x = x_min
    for i in range(int(segments)):
        left_x = accum_x
        accum_x = accum_x + step_size
        if accum_x <= x_max:
            right_x = accum_x
        else:
            right_x = x_max
        x_mid.append(1 / 2 * (left_x + right_x))  
        # check whether the data point of x is in the segments range 
        x_in_range = x_data.apply(lambda t: left_x <= t <= right_x)
        # find the corresponding data point in y
        y_in_range = y_data[x_in_range]
        # make sure that when n=1, std()'s result will not be nan 
        y_in_range_mean = y_in_range.mean()
        y_in_range_std = y_in_range.std()
        print("Segment:", i)
        print("Mean:", y_in_range_mean)
        print("Std Error:", y_in_range_std)
        # append the values to the list
        if pd.isnull(y_in_range_mean):
            # replace nan with 0
            moving_mean.append(0)
        else:
            moving_mean.append(y_in_range_mean)
        if pd.isnull(y_in_range_std):
            # replace nan with 0
            moving_std.append(0)
        else:
            moving_std.append(y_in_range_std)
    
    return moving_mean, moving_std, x_mid

def moving_average(step_size, data_name):
    """calculate moving mean and moving standard deviation value
    :param step_size: how wide every interval has
    :param data_name: the file stores data
    """
    data = pd.read_excel(data_name, engine="openpyxl")
    # set X-Y pairs 
    x = data.loc[:, 'Moho': '300km']
    y = data.loc[:, 'N': 'M']
    x_column, y_column = x.columns, y.columns
    print("Moving Average Calculation ...")
    print("Step Size:", step_size)
    print("The number of pair X-Y:", len(x_column) * len(y_column))
    for single_x in x_column:
        store = {}    
        for single_y in y_column:
            print("X selected:", single_x)
            print("Y selected:", single_y)
            moving_mean, moving_std, x_mid = single_x_vs_y(x[single_x], y[single_y], step_size)
            
            # store the results in a dictionary
            pair = str(single_x) + '-' + str(single_y)
            key1, key2, key3 = "moving_mean", "moving_std", "x_mid"
            data = dict([[key1, moving_mean], [key2, moving_std], [key3, x_mid]])
            if pair in store:
                store[pair].append(data)
            else:    
                store[pair] = []
                store[pair].append(data)
             
            print("----- Result  -----")
            print("Step Size:", step_size) 
            print("Moving Mean:", moving_mean)
            print("Moving Std Deviation:", moving_std)
            print("  ")
        pd.DataFrame(store).to_excel(os.path.join(RESULT_PATH, "{}.xlsx".format(single_x)))

if __name__ == "__main__":
    moving_average(1, "MAJOR3P.xlsx")
