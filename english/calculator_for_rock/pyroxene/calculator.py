#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import time

def find_num(major_el_name):
    """
    Find the number of cations and the number of oxygen atoms of the principal element in the listing

    :param major_el_name: Listing of principal elements
    :return: Number of cations and number of oxygen atoms
    """
    length = len(major_el_name)
    temp_ion_num = [re.findall('\d?O', major_el_name[i], re.I) for i in range(length)]
    ion_num = []
    for i in range(length):
        ion_num.extend(temp_ion_num[i])
    for j in range(length):
        ion_num[j] = re.findall('\d*', ion_num[j])[0]
        if ion_num[j] == '':
            ion_num[j] = 1
        else:
            ion_num[j] = int(ion_num[j])

    temp_oxy_num = [re.findall('O\d?', major_el_name[i], re.I) for i in range(length)]
    oxy_num = []
    for i in range(length):
        oxy_num.extend(temp_oxy_num[i])
    for j in range(length):
        oxy_num[j] = re.findall('\d*', oxy_num[j])[1]
        if oxy_num[j] == '':
            oxy_num[j] = 1
        else:
            oxy_num[j] = int(oxy_num[j])
    return ion_num, oxy_num

def find_ion(major_el_name):
    """
    Find the cation in the principal element of the listing

    :param major_el_name: The name of the main element column
    :return:  cations
    """
    length = len(major_el_name)
    temp = []
    for i in range(length):
        a = re.findall('[a-zA-Z]{1,2}[\d*]?', major_el_name[i], re.I)
        temp.append(a[0])
    ion = []
    for i in range(length):
        ion.extend(re.findall('[a-zA-Z]{1,2}', temp[i], re.I))
    return ion

def rel_mole_weight(ion, ion_num, oxy_num):
    """
    Calculating Relative Molecular Weight

    :param ion: Each cation
    :param ion_num: Number of cations per cation
    :param oxy_num: The number of oxygen atoms corresponding to each cation
    :return: Relative molecular weight
    """
    ion_dict = {'Si':28.085, 'Ti':47.867, 'Al':26.981, 'Cr':51.996, 'Fe':55.845, 'Mn':54.938,
                'Mg':24.305, 'Ca':40.078, 'Na':22.989, 'K':39.098, 'P':30.974, 'Ni':58.693,
                'Zn':65.390, 'Li':6.941, 'Zr':91.224, 'V':50.941, 'O':15.999}
    length = len(ion)
    if length != len(ion_num) or length != len(oxy_num):
        raise Exception

    relative_molecular_weight = []
    for i in range(length):
        a = ion_dict[ion[i]] * ion_num[i] + ion_dict['O'] * oxy_num[i]
        relative_molecular_weight.append(a)
    return relative_molecular_weight

def conver_ratio(rmw, oxy_num, mf):
    """
    Calculation of conversion factors

    :param rmw: Relative molecular weight
    :param mf: Mass fraction of the principal element
    :return: Value of the conversion factor
    """
    conversion_ratio = float(6) / sum(np.array(oxy_num) * np.array(mf) / np.array(rmw))
    return conversion_ratio

def output(cr, rmw, ion_num, mf):
    '''
    Calculate the output y for each cation

    :param cr: conversion factor
    :param rmw: Relative molecular weight
    :param ion_num: Number of cations
    :param mf: Mass fraction of the principal element
    :return: Output y of each cation
    '''
    y = cr * np.array(mf) * np.array(ion_num) / np.array(rmw)
    return y

def projection(index, target, y):
    '''
    Calculation of the projection value of a specific cation in the range of 0 to 1

    :param index: Index to the specified cation list
    :param target: List of specified cations
    :param y: Output value of each cation y
    :return: Projected values of specific cations
    '''
    sum = 0
    for i in range(len(target)):
        sum += np.array(y[target[i]])
    # sum = np.array(y[target[0]]) + np.array(y[target[1]]) + np.array(y[target[2]])
    proj = np.array(y[target[index]]) / sum
    return proj


def main():
    start_time = time.time()
    print("读取文件............")
    data = pd.read_excel('cal_data_4th.xlsx')         # Read the data set
    data.fillna(0, inplace=True)                      # The interpolation value is zero, there can be no null value

    data_columns = list(data.columns)
    # print("列名：", data_columns)                    # Listing name: principal element

    ion_num, oxy_num = find_num(data_columns)
    ion = find_ion(data_columns)
    # print("阳离子: ", ion)                           # Demonstrate cation
    # print("阳离子个数: ", ion_num)                    # Number of cations
    # print("氧原子个数: ",oxy_num)                     # Number of oxygen atoms
    # print("维度:", len(ion), len(ion_num), len(oxy_num)) # Compare whether the latitudes are the same

    rmw = rel_mole_weight(ion, ion_num, oxy_num)
    # print("相对分子质量:", np.array(rmw))             # Relative molecular weight
    cr_columns = []
    data_num = data.shape[0]
    for i in range(data_num):
        a = data.iloc[i, :]
        cr = conver_ratio(rmw, oxy_num, a)            # Calculation of conversion factors
        cr_columns.append(cr)                         # Preservation of conversion factors

    temp = []
    for j in range(data_num):
        b = data.iloc[j, :]
        y = output(cr_columns[j], rmw, ion_num, b)    # Calculate the output value y for each cation
        temp.append(y)                                # Save output value y
    temp_df = pd.DataFrame(temp)                      # New DataFrame table to save the output value y
    temp_df.columns = ion                             # Adds a column name to the DataFrame table with output value y
    # print(temp_df)
    data['换算系数'] = np.array(cr_columns).reshape(-1, 1)  # Add a new column [conversion factor] to the original data set [data]
    # print(data['换算系数'])
    # print(data)                                     # Original data set with conversion factors
    new_df = pd.concat([data, temp_df], axis=1)       # Merge the DataFrame table of the original dataset with the DataFrame table of the output value y
    # print(new_df)                                   # Data set containing conversion coefficients and y columns of output values for each cation

    target = ['Fe', 'Mg', 'Ca']                       # Selected cations to be projected
    df1 = new_df[target]
    target_list = []
    for i in range(data_num):
        y = df1.iloc[i, :]
        ls = []
        for j in range(len(target)):
            proj = projection(j, target, y)           # Calculation of the projected value of a given cation
            ls.append(proj)                           # Save projection values
        #print(ls)
        target_list.append(ls)
    target_df = pd.DataFrame(target_list)             # New DataFrame table to save projected values
    # print(pd.DataFrame(target_list))
    project_name = [target[i] + '_projected' for i in range(len(target))]   # Constructing new listings with projected values
    target_df.columns = project_name                                        # Adds a column name to a DataFrame table that holds the projected values
    final_df = pd.concat([new_df, target_df], axis=1)  # Combination of raw data tables with conversion factors and output values and DF tables with stored projection values
    # print(final_df)                                  # The final form we'll need

    final_df.to_csv("new_cal_data_4th.csv")            # Save the final table as a csv file

    end_time = time.time()
    print("程序运行时间:{}s".format(end_time-start_time))
    
    
if __name__ == '__main__':
    main()



