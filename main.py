# 导入包
import os
import random
import shutil
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
from tslearn.utils import to_time_series_dataset
from tslearn.clustering import KShape, TimeSeriesKMeans, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import silhouette_score
from sklearn.metrics import calinski_harabasz_score

'''
任务数据集仅包含患者用药量单维度的时序信息，因此任务总结为单变量时序聚类问题。
'''

# 自定义时序聚类的类
class TimeSeriesCluster():
    def __init__(self, opt):
        '''
        初始化
        :param opt: 用户指定参数
        '''
        # 用户指定参数
        self.opt = opt
        # 算法名称
        self.name = {'dtw_kmeans':'DTW KMeans', 'soft_dtw_kmeans':'Soft-DTW KMeans', 'gak_kernal_kmeans':'GAK kernal-KMeans', 'k_shape':'k-Shape'}
        # 设置绘图风格
        sns.set(style='darkgrid', font_scale=1.5)
        # 绘图颜色列表
        self.color_list = list(mcd.XKCD_COLORS.values())
        # 清空输出目录
        if not os.path.exists('imgs'):
            os.mkdir('imgs')
        else:
            shutil.rmtree('imgs')
            os.mkdir('imgs')
        # 如果使用k-Shape聚类算法，则应该进行数据标准化
        if opt.mode == 'k_shape':
            print('使用k-Shape聚类算法，将进行数据标准化。')
            self.opt.scale = True

    # 绘制数据集时序数据折线图
    def plot_data(self, ori_data, data, file_name, colors):
        '''
        绘制数据集时序数据折线图
        :param ori_data: 原始数据集
        :param data: 标准时序数据格式的数据集
        :param file_name: 输出图片文件名称
        :param colors: 折线颜色列表
        :return: 输出时序数据折线图的图片文件
        '''
        # 获取日期
        date = ori_data.columns[1:]
        # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
        plt.figure(figsize=(16, 10), dpi=(80))
        # 绘制所有时序数据的折线图
        for i in range(len(data)):
            plt.plot(data[i].ravel(), color=colors[i])
        # 设置x轴刻度为日期，旋转30度，字体大小为12
        plt.xticks(range(0, len(date), 3), date[::3], rotation=30, fontsize=12)
        # 设置y轴刻度字体大小为12
        plt.yticks(fontsize=12)
        # 设置标题，字体大小为24
        plt.title(file_name, fontsize=24)
        # 设置x轴、y轴名称，字体大小为20
        plt.xlabel('date', fontsize=20)
        plt.ylabel('dosage', fontsize=20)
        # 保存图片
        plt.savefig(os.path.join('imgs', '{}.png'.format(file_name)))
        # 展示图片
        plt.show()

    # 获取模型输入数据并绘制时序数据折线图
    def get_input_data(self):
        '''
        读取原始数据集，并转换为标准时序数据结构，对数据进行z-score标准化得到算法输入数据
        :return: 原始数据集、标准化后的算法输入数据集
        '''
        # 导入患者用药剂量原始数据。
        ori_data = pd.read_csv(self.opt.dataset)
        # 转换为标准时序数据格式，从原始结构(2维数组M×N，行表示患者，列表示日期)转换为标准时序数据结构(3维数组M×N×1，第0维表示患者，第1维表示日期，2维表示用药量(因为只有用药量这个变量，所以此维度的长度是1))
        data = to_time_series_dataset(ori_data.iloc[:, 1:])
        if self.opt.scale:
            # 数据标准化，具体对每个患者用药量各自进行z-score标准化，使每个患者用药量的均值为0，标准差为1
            input_data = TimeSeriesScalerMeanVariance(mu=0., std=1.).fit_transform(data)
        else:
            # 不标准化
            input_data = data
        # 设置颜色
        colors = random.sample(self.color_list, len(data))
        # 原始数据可视化
        self.plot_data(ori_data, data, 'Ori Dataset', colors)
        # 标准化后数据可视化
        if self.opt.scale:
            self.plot_data(ori_data, input_data, 'Scale Dataset', colors)
        return ori_data, input_data

    # 时序聚类算法
    def ts_cluster(self, X, n_clusters, mode='k-shape', max_iter=500, n_init=50, seed=0):
        '''
        时序聚类算法包括:
        1.'dtw_kmeans':基于DTW距离的KMeans聚类算法
        2.'soft_dtw_kmeans':基于soft-DTW距离的KMeans聚类算法
        3.'gak_kernal_kmeans':基于GAK核的核KMeans聚类算法
        4.'k_shape':KShape聚类算法
        :param X:输入数据
        :param n_clusters:类簇数目
        :param mode:聚类模式，即指定所选择的聚类算法，包括:'dtw_kmeans'、'soft_dtw_kmeans'、'gak_kernal_kmeans'、'k_shape'，默认值为'k-shape'
        :param max_iter:聚类算法最大迭代次数，默认值为500
        :param n_init:尝试多少次不同随机种子进行聚类，获取其中最好的结果，默认值为50
        :param seed:随机种子，默认值为0
        :return:数据聚类的标签结果数组、类内距离平方和、聚类中心组成的字典，其中算法'dtw_kmeans'、'soft_dtw_kmeans'、'k_shape'才支持输出聚类中心
            {
            'label':聚类标签结果数组,
            'inertia':类内距离平方和,
            'cluster_centers':聚类中心
            }。
        '''
        # 聚类算法对象
        cluster_algorithms = {'dtw_kmeans':TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=max_iter, n_init=n_init, verbose=True, random_state=seed),
                              'soft_dtw_kmeans':TimeSeriesKMeans(n_clusters=n_clusters, metric="softdtw", max_iter=max_iter, n_init=n_init, metric_params={"gamma": .01}, verbose=True, random_state=seed),
                              'gak_kernal_kmeans':KernelKMeans(n_clusters=n_clusters, kernel="gak", kernel_params={"sigma": "auto"}, max_iter=max_iter, n_init=n_init, verbose=True, random_state=seed),
                              'k_shape':KShape(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, verbose=True, random_state=seed)}
        # 进行聚类，并返回对应的类别标签、类内距离平方和、聚类中心
        model = cluster_algorithms[mode]
        label = model.fit_predict(X)
        result = {'label':label,
                  'inertia':model.inertia_,
                  'cluster_centers':model.cluster_centers_ if mode in ['dtw_kmeans', 'soft_dtw_kmeans', 'k_shape'] else None}
        return result

    # 肘部法则确定最佳类簇数目
    def plot_elbow(self, n_result, name, max_n_clusters):
        '''
        利用肘部法则确定最佳类簇数目，并绘制肘部图。
        :param n_result:各类簇数目进行聚类的结果字典
        :param name:聚类算法名称
        :param max_n_clusters: 类簇数目最大值
        :return:输出肘部图文件，返回最佳类簇数目
        '''
        # 获取类簇数目序列，类簇数目从1开始聚类(因为要能选到类簇数目为2)，直到聚类到最大类簇数目+1(因为要能选到最大类簇数目)
        n_list = range(1, max_n_clusters + 2)
        # 获取各类簇数目聚类得到的类内距离平方和序列
        inertia_list = [n_result[n]['inertia'] for n in n_list]
        # 1.绘制肘部图
        # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
        plt.figure(figsize=(12, 10), dpi=(80))
        # 绘制折线图，具体为红色、实心点、实线，线粗3个像素
        plt.plot(n_list, inertia_list, 'ro-', linewidth=3)
        # 设置标题
        plt.title('{} Elbow'.format(self.name[name]), fontsize=24)
        # 设置x轴、y轴名称
        plt.xlabel('number of clusters', fontsize=20)
        plt.ylabel('inertia', fontsize=20)
        # 保存图片
        plt.savefig(os.path.join('imgs', '{} Elbow.png'.format(self.name[name])))
        # 显示图片
        plt.show()
        # 2.获取最佳类簇数目
        # 计算类内距离平方和序列一阶差分
        diff1 = np.diff(inertia_list)
        # 相邻类簇数目的类内距离平方和相除，获取比例，获取最大比例的索引值，根据该索引值+1访问类簇数目序列，得到最佳类簇数目。
        best_n_clusters = n_list[np.argmax(diff1[:-1] / diff1[1:]) + 1]
        return best_n_clusters

    # 轮廓系数确定最佳类簇数目
    def plot_silhouette_score(self, X, n_result, name, max_n_clusters, metric='dtw'):
        '''
        利用轮廓系数确定最佳聚类数目，并绘制各个类簇数目对应的轮廓系数折线图。
        :param X: 算法输入数据集
        :param n_result: 各类簇数目进行聚类的结果字典
        :param name: 聚类算法名称
        :param max_n_clusters: 类簇数目最大值
        :param metric: 计算轮廓系数时所使用的度量方法
        :return: 输出轮廓系数折线图文件，返回最佳类簇数目
        '''
        # 获取类簇数目序列，类簇数目从2开始聚类(因为选类簇数目为1没意义)，直到聚类到最大类簇数目
        n_list = range(2, max_n_clusters + 1)
        # 计算各类簇数目聚类结果的轮廓系数序列
        silhouette_score_list = [silhouette_score(X, n_result[n]['label'], metric=metric) for n in n_list]
        # 1.绘制轮廓系数曲线图
        # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
        plt.figure(figsize=(12, 10), dpi=(80))
        # 绘制折线图，具体为红色、实心点、实线，线粗3个像素
        plt.plot(n_list, silhouette_score_list, 'ro-', linewidth=3)
        # 设置标题
        plt.title('{} Silhouette Score'.format(self.name[name]), fontsize=24)
        # 设置x轴、y轴名称
        plt.xlabel('number of clusters', fontsize=20)
        plt.ylabel('silhouette score', fontsize=20)
        # 保存图片
        plt.savefig(os.path.join('imgs', '{} Silhouette Score.png'.format(self.name[name])))
        # 显示图片
        plt.show()
        # 2.获取最佳类簇数目
        # 当轮廓系数最大时，聚类效果最好，因此获取最大轮廓系数对应的类簇数目作为最佳类簇数目
        best_n_clusters = n_list[np.argmax(silhouette_score_list)]
        return best_n_clusters

    # calinski_harabasz指标确定最佳类簇数目
    def plot_calinski_harabasz_score(self, X, n_result, name, max_n_clusters):
        '''
        利用calinski_harabasz指标确定最佳聚类数目，并绘制各个类簇数目对应的calinski_harabasz指标折线图。
        :param X: 算法输入数据集
        :param n_result: 各类簇数目进行聚类的结果字典
        :param name: 聚类算法名称
        :param max_n_clusters: 类簇数目最大值
        :return: 输出calinski_harabasz指标折线图文件，返回最佳类簇数目
        '''
        # 获取类簇数目序列，类簇数目从2开始聚类(因为选类簇数目为1没意义)，直到聚类到最大类簇数目
        n_list = range(2, max_n_clusters + 1)
        # 计算各类簇数目聚类结果的calinski_harabasz指标序列
        calinski_harabasz_list = [calinski_harabasz_score(X.squeeze(), n_result[n]['label']) for n in n_list]
        # 1.绘制calinski harabasz指标曲线图
        # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
        plt.figure(figsize=(12, 10), dpi=(80))
        # 绘制折线图，具体为红色、实心点、实线，线粗3个像素
        plt.plot(n_list, calinski_harabasz_list, 'ro-', linewidth=3)
        # 设置标题
        plt.title('{} Calinski Harabasz Score'.format(self.name[name]), fontsize=24)
        # 设置x轴、y轴名称
        plt.xlabel('number of clusters', fontsize=20)
        plt.ylabel('calinski harabasz score', fontsize=20)
        # 保存图片
        plt.savefig(os.path.join('imgs', '{} Calinski Harabasz Score.png'.format(self.name[name])))
        # 显示图片
        plt.show()
        # 2.获取最佳类簇数目
        # 当calinski harabasz指标值最大时，聚类效果最好，因此获取最大calinski harabasz指标对应的类簇数目作为最佳类簇数目
        best_n_clusters = n_list[np.argmax(calinski_harabasz_list)]
        return best_n_clusters

    # 获取最佳类簇数目
    def get_best_n(self, X):
        '''
        确定最佳类簇数目。
        :param X: 算法输入数据集
        :return: 最佳类簇数目
        '''
        # 依次指定不同类簇数目进行聚类，
        n_result = {n_clusters: self.ts_cluster(X, n_clusters, mode=self.opt.mode, max_iter=self.opt.max_iter, n_init=self.opt.n_init, seed=self.opt.seed) for n_clusters in range(1, self.opt.max_n_clusters + 2)}
        # 最佳类簇数目字典
        best_n_clusters_dict = {'elbow':self.plot_elbow(n_result, self.opt.mode, self.opt.max_n_clusters),
                                'silhouette':self.plot_silhouette_score(X, n_result, self.opt.mode, self.opt.max_n_clusters, metric='dtw'),
                                'calinski_harabasz':self.plot_calinski_harabasz_score(X, n_result, self.opt.mode, self.opt.max_n_clusters)}
        return best_n_clusters_dict[self.opt.n_mode]

    # 绘制最佳类簇数目聚类结果并对原始数据打标签
    def plot_cluster_result(self, ori_data, X, best_n_clusters):
        '''
        根据最佳类簇数目进行聚类，并绘制时序数据聚类结果折线图，输出打标后的数据集。
        :param ori_data:原始数据集
        :param X:算法输入数据集
        :param best_n_clusters:最佳类簇数目
        :return:输出聚类后各个标签对应的时序数据折线图、所有标签的时序数据折线图、聚类结果csv文件
        '''
        # 根据最优类簇数目聚类
        result = self.ts_cluster(X, best_n_clusters, mode=self.opt.mode, max_iter=self.opt.max_iter, n_init=self.opt.n_init, seed=self.opt.seed)
        labels = result['label']
        # 获取日期
        date = ori_data.columns[1:]
        # 各类别颜色列表
        # colors = random.sample(self.color_list, best_n_clusters)
        colors = ['#50c48f','#f5616f','#0780cf','#f9e264','#457b9d'] #绿，红，黄,lan]
        # 1.按类别标签分别绘制时序折线图
        for l in range(best_n_clusters):
            # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
            plt.figure(figsize=(16, 10), dpi=(80))
            # 绘制某一标签的所有折线，指定随机颜色，透明度为0.5
            for x in X[labels==l]:
                plt.plot(x.ravel(), color=colors[l], alpha=0.5)
            # 绘制该类的聚类中心
            if result['cluster_centers'] is not None:
                plt.plot(result['cluster_centers'][l].ravel(), color=colors[l], alpha=0.8, linewidth=10)
            # 设置x轴刻度为日期，旋转30度，字体大小为12
            plt.xticks(range(0, len(date), 3), date[::3], rotation=30, fontsize=12)
            # 设置y轴刻度字体大小为12
            plt.yticks(fontsize=12)
            # 设置标题，字体大小为24
            plt.title('{} Cluster Result of Label {}'.format(self.name[self.opt.mode], l), fontsize=24)
            # 设置x轴、y轴名称，字体大小为20
            plt.xlabel('date', fontsize=20)
            plt.ylabel('dosage', fontsize=20)
            # 保存图片
            plt.savefig(os.path.join('imgs', 'Label{}.png'.format(l)))
            # 显示图片
            plt.show()
        # 2.所有类别时序折线绘制到同一幅图中
        # 设置图像大小，figsize=(宽,长)，单位:英寸，dpi=图像分辨率
        plt.figure(figsize=(16, 10), dpi=(80))
        for l in range(best_n_clusters):
            # 绘制某一标签的所有折线，指定随机颜色，透明度为0.5
            for i, x in enumerate(X[labels==l]):
                if i == 0 and result['cluster_centers'] is None:
                    plt.plot(x.ravel(), color=colors[l], alpha=0.5, label='label{}'.format(l))
                    continue
                plt.plot(x.ravel(), color=colors[l], alpha=0.5)
            # 绘制该类的聚类中心
            if result['cluster_centers'] is not None:
                plt.plot(result['cluster_centers'][l].ravel(), color=colors[l], alpha=0.8, linewidth=10, label='label{}'.format(l))
        # 设置x轴刻度为日期，旋转30度，字体大小为12
        plt.xticks(range(0, len(date), 3), date[::3], rotation=30, fontsize=12)
        # 设置y轴刻度字体大小为12
        plt.yticks(fontsize=12)
        # 设置标题，字体大小为24
        plt.title('{} Cluster Result of All Label'.format(self.name[self.opt.mode]), fontsize=24)
        # 设置x轴、y轴名称，字体大小为20
        plt.xlabel('date', fontsize=20)
        plt.ylabel('dosage', fontsize=20)
        # 设置图例
        plt.legend(fancybox=True, loc='best')
        # 保存图片
        plt.savefig(os.path.join('imgs', 'All Label.png'))
        # 显示图片
        plt.show()
        # 3.对原始数据集打标
        ori_data['label'] = labels
        # 输出带标签的数据集csv文件
        ori_data.to_csv('result.csv')


# 用户指定参数，从终端调用脚本时可以直接传参
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data.csv', help='原始数据集文件路径')
    parser.add_argument('--mode', type=str, choices=['dtw_kmeans', 'soft_dtw_kmeans', 'gak_kernal_kmeans', 'k_shape'], default='dtw_kmeans', help='时序聚类算法名称，默认使用基于DTW的KMeans算法')
    parser.add_argument('--n_mode', type=str, choices=['elbow', 'silhouette', 'calinski_harabasz'], default='silhouette', help='获取最佳类簇数目n的策略，默认使用轮廓系数')
    parser.add_argument('--max_iter', type=int, default=100, help='聚类算法最大迭代次数，默认值为100')
    parser.add_argument('--n_init', type=int, default=30, help='尝试多少个不同随机种子进行聚类，获取其中最好的结果，默认值为30')
    parser.add_argument('--max_n_clusters', type=int, default=10, help='类簇数目最大值，默认值为10')
    parser.add_argument('--seed', type=int, default=1, help='聚类算法使用的随机种子，默认值为1')
    parser.add_argument('--scale', action='store_true', default=True, help='数据是否进行标准化，默认值为False')
    opt = parser.parse_args()
    return opt


# 主函数
def main():
    # 获取用户指定参数
    opt = parse_opt()
    # 实例化对象
    tscluster = TimeSeriesCluster(opt)
    # 获取模型输入数据
    ori_data, input_data = tscluster.get_input_data()
    # 获取最佳类簇数目
    best_n_clusters = tscluster.get_best_n(input_data)
    print('最优类簇数目为:{}'.format(best_n_clusters))
    # 绘制最佳类簇数目聚类结果
    tscluster.plot_cluster_result(ori_data, input_data, best_n_clusters)


if __name__ == '__main__':
    main()