#coding = utf-8
# Author: Hu Baitao
#Function : app on browser drawing 3d graph
#Date : 2020-08-27
#zhuliang3000.xlsx为案例表


import xlrd
import plotly.graph_objs as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output, State
import pandas as pd


#输入函数
def para_input():
    print('请输入数据文件路径,.xlsx文件')    #D:\homework\Research 2020 S\10 cycle old alg\zhuliang3000.xlsx
    filename = input()

    return filename

#读取表格，处理数据
def read_xlsx(filename):
    # file_object = xlrd.open_workbook(filename)
    # sheetnames = file_object.sheet_names()
    # sheetwork = file_object.sheet_by_name(sheetnames[0])
    # nrows = sheetwork.nrows
    # ncols = sheetwork.ncols
    # data = []
    # data_title = []
    #
    # for i in range(ncols):
    #     data_title.append(sheetwork.cell_value(0,i))
    # data.append(data_title)
    #
    # for j in range(ncols):
    #     new_row_data = []
    #     for k in range(1,nrows):
    #         new_row_data.append(sheetwork.cell_value(k,j))
    #     data.append(new_row_data)
    data =  pd.read_excel(filename)

    return data

# 展示数据
def generate_table(dataframe, max_rows):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])

def generate_graph(dataframe, x_value, y_value, z_value):
    trace = go.Scatter3d(
        x=dataframe[x_value], y=dataframe[y_value], z=dataframe[z_value], mode='markers', marker=dict(
            size=5,
            color=dataframe[z_value],  # set color to an array/list of desired values
            colorscale='Viridis'
        )
    )

    layout = go.Layout(title='主量元素分析',
                       scene=dict(
                           xaxis_title=x_value,
                           yaxis_title=y_value,
                           zaxis_title=z_value
                       ),
                       height= 800,
                       width= 1000
    )
    fig = go.Figure(data=[trace], layout=layout)

    return fig

def main():
    filename = para_input()
    df = read_xlsx(filename)

    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(children=[
        html.H1(
            children='3D Graph',
            style = {
                'textAlign': 'center',
                # 'color': colors['text']
                }
        ),

        html.H4(children='主量3000'),

        dcc.Dropdown(
            id='num_row',
            options=[{'label': 'show first 10 rows', 'value': 10},
                     {'label': 'show first 25 rows', 'value': 25},
                     {'label': 'show first 50 rows', 'value': 50},
                     {'label': 'show first 100 rows', 'value': 100}],
            value=10
        ),

        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            page_size =10,
        ),

        html.Label('选择三个主元素'),
        dcc.Checklist(
            id='box-section',
            options=[
                {'label': 'TRUE VALUE', 'value': 'TRUE VALUE'},
                {'label': 'SIO2(WT%)', 'value': 'SIO2(WT%)'},
                {'label': 'TIO2(WT%)', 'value': 'TIO2(WT%)'},
                {'label': 'AL2O3(WT%)', 'value': 'AL2O3(WT%)'},
                {'label': 'CR2O3(WT%)', 'value': 'CR2O3(WT%)'},
                {'label': 'FEOT(WT%)', 'value': 'FEOT(WT%)'},
                {'label': 'CAO(WT%)', 'value': 'CAO(WT%)'},
                {'label': 'MGO(WT%)', 'value': 'MGO(WT%)'},
                {'label': 'MNO(WT%)', 'value': 'MNO(WT%)'},
                {'label': 'K2O(WT%)', 'value': 'K2O(WT%)'},
                {'label': 'NA2O(WT%)', 'value': 'NA2O(WT%)'}
            ],
            value=['TRUE VALUE', 'SIO2(WT%)','TIO2(WT%)']
        ),

        html.Button(id='submit-button-state', n_clicks=0,children='Submit'),

        dcc.Graph(
            id='graph with main element',
            figure= generate_graph(df,'TRUE VALUE','SIO2(WT%)','TIO2(WT%)')
        )
    ])

    @app.callback(
        Output('table','page_size'),
        [Input('num_row', 'value')])
    def update_row_num(row_num):
        return row_num


    @app.callback(
        Output('graph with main element', 'figure'),
        [Input('submit-button-state', 'n_clicks')],
        [State('box-section', 'value')])
    def update_figure(n_clicks, box_value):
        fig = generate_graph(df, box_value[0], box_value[1], box_value[2])

        return fig

    app.run_server(debug=True)

if __name__=='__main__':
    main()