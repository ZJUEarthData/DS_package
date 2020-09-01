# -*- coding: utf-8 -*-

class ElementsInCurve:

    def __init__(self,filename_1, filename_2,sheet_name):
        """
        Input the file containing the elements data
        :param filename_1: 微量全700+数据
        :param filename_2: trace标准化值(ppm)
        :param sheet_name: 0 = 稀土元素; 1 = 微量多元素
        """
        import pandas as pd

        raw_data = pd.read_excel(filename_1, sheet_name)
        self.df1 = raw_data[raw_data["是否交代"] == 1].drop(["是否交代", "CITATION"], axis=1)
        self.df2 = raw_data[raw_data["是否交代"] == -1].drop(["是否交代", "CITATION"], axis=1)
        standard_data = pd.read_excel(filename_2, sheet_name, header=1, index="Element")
        # ppm to ppb
        self.std = standard_data.drop(["Element"], axis=1) * 1000

    def plot(self, x_index = None,
             num_data_1 = None, num_data_2 = None,
             fig_size = (12,8),
             save_png = None):
        """
        plot the elements in curve
        :param x_index: X轴的元素坐标（默认使用全部元素,大写）
        :param num_data_1: 使用（是否交代=1）数据的数量（默认使用全部数据）
        :param num_data_2: 使用（是否交代=-1）数据的数量（默认使用全部数据）
        :param fig_size: 图片大小，默认为（12，8）
        :param save_png: 保存图片的文件名， 默认不存储图片
        :return: 展示图片(或另存为)
        """
        import matplotlib.pyplot as plt
        from collections import OrderedDict

        if num_data_1 == None:
            num_data_1 = len(self.df1)
        if num_data_2 == None:
            num_data_2 = len(self.df2)

        if x_index == None:
            x_index = list(self.std.columns)
            x_index = [x.upper() for x in x_index]
            self.std.columns = [x.upper() for x in x_index]
            x_index = list(set(x_index) & set(self.df1.columns))

        data1 = self.df1[x_index]
        data2 = self.df2[x_index]
        self.std = self.std[x_index]
        result1 = data1.div(self.std.iloc[0].values)
        result2 = data2.div(self.std.iloc[0].values)

        plt.figure(figsize=fig_size)
        for i in range(num_data_1):
            plt.semilogy(range(len(result1.columns.values)), result1.iloc[i], 'b.-', label=1)
            plt.xticks(range(len(result1.columns.values)), result1.columns.values)
        for j in range(num_data_2):
            plt.semilogy(range(len(result2.columns.values)), result2.iloc[j], 'k.-', label=-1)
            plt.xticks(range(len(result2.columns.values)), result2.columns.values)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        if save_png != None:
            plt.savefig(save_png)
        plt.show()


def main():
    # 微量多元素标准化图解
    EN_multi = ElementsInCurve('微量全700+数据（ppb）.xlsx', 'trace标准化值(ppm).xlsx', 1)
    EN_multi.plot(save_png = '微量多元素标准化图解.png')
    # 稀土元素蛛网图
    EN_rare = ElementsInCurve('微量全700+数据（ppb）.xlsx', 'trace标准化值(ppm).xlsx', 0)
    EN_rare.plot(save_png='稀土元素蛛网图.png')
    # 选取微量元素
    # x_index = ['LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU']
    # EN_rare.plot(x_index = x_index)
    # 选取100个样本
    # EN_rare.plot(num_data_1 = 100,num_data_2 = 100)


if __name__ == '__main__':
    main()