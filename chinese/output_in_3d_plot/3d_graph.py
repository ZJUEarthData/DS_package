#coding = utf-8
# Author: Hu Baitao
#Function : draw a 3D graph on browser
#Date : 2020-08-26
#zhuliang3000.xlsx为案例表


import xlrd
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"

#输入函数
def para_input():
    print('请输入数据文件路径,.xlsx文件')    #D:\homework\Research 2020 S\10 cycle old alg\zhuliang3000.xlsx
    filename = input()

    return filename

#读取表格，处理数据
def read_xlsx(filename):
    file_object = xlrd.open_workbook(filename)
    sheetnames = file_object.sheet_names()
    sheetwork = file_object.sheet_by_name(sheetnames[0])
    nrows = sheetwork.nrows
    ncols = sheetwork.ncols
    data = []
    data_title = []

    for i in range(ncols):
        data_title.append(sheetwork.cell_value(0,i))
    data.append(data_title)

    for j in range(ncols):
        new_row_data = []
        for k in range(1,nrows):
            new_row_data.append(sheetwork.cell_value(k,j))
        data.append(new_row_data)

    return data


#主函数
def main():
    filename = para_input()
    data = read_xlsx(filename)

    x = data[4]
    y = data[6]
    z = data[11]
    trace = go.Scatter3d(
        x=x, y=y, z=z, mode='markers', marker=dict(
            size=5,
            color=z,  # set color to an array/list of desired values
            colorscale='Viridis'
        )
    )

    layout = go.Layout(title='主量元素分析',
                       scene=dict(
                           xaxis_title='AL2O3(WT%)',
                           yaxis_title='FEOT(WT%)',
                           zaxis_title='NA2O(WT%)'
                       ))

    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

if __name__=='__main__':
    main()