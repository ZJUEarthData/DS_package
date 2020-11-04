# -*- coding: utf-8 -*-

class ElementsInCurve:

    def __init__(self,filename_1, filename_2,sheet_name):
        """
        Input the file containing the elements data
        :param filename_1: Trace elements total 700 + data
        :param filename_2: trace Standardized values (ppm)
        :param sheet_name: 0 = Rare earth elements; 1 = Trace multi element
        """
        import pandas as pd

        raw_data = pd.read_excel(filename_1, sheet_name)
        self.df1 = raw_data[raw_data[" Whether or not metasomatism"] == 1].drop(["Whether or not metasomatism", "CITATION"], axis=1)
        self.df2 = raw_data[raw_data["Whether or not metasomatism"] == -1].drop(["Whether or not metasomatism", "CITATION"], axis=1)
        standard_data = pd.read_excel(filename_2, sheet_name, header=1, index="Element")
        # ppm to ppb
        self.std = standard_data.drop(["Element"], axis=1) * 1000

    def plot(self, x_index = None,
             num_data_1 = None, num_data_2 = None,
             fig_size = (12,8),
             save_png = None):
        """
        plot the elements in curve
        :param x_index: X-axis element coordinates (default all elements, uppercase)
        :param num_data_1: The amount of data used (Whether or not metasomatism =1) (all data is used by default)
        :param num_data_2: The amount of data used (Whether or not metasomatism =-1) (the default is to use all data)
        :param fig_size: Image size, default: (12,8)
        :param save_png: Save the file name of the image. By default, the image is not stored
        :return: Display picture (or save as)
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
    # Trace multi - element standardization diagram
    EN_multi = ElementsInCurve('Trace elements total 700 + data（ppb）.xlsx', 'trace Standardized values(ppm).xlsx', 1)
    EN_multi.plot(save_png = 'Trace multi - element standardization diagram.png')
    # Rare earth element spider diagram
    EN_rare = ElementsInCurve('Trace elements total 700 + data（ppb）.xlsx', 'trace Standardized values(ppm).xlsx', 0)
    EN_rare.plot(save_png='Rare earth element spider diagram.png')
    # Select trace elements
    # x_index = ['LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU']
    # EN_rare.plot(x_index = x_index)
    # pick 100 samples
    # EN_rare.plot(num_data_1 = 100,num_data_2 = 100)


if __name__ == '__main__':
    main()