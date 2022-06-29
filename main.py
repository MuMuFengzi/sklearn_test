import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import jieba
import pandas as pd


def datasets_demo():
    """
    sklearn 数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print('鸢尾花数据集:\n', iris)
    print("鸢尾花描述\n", iris["DESCR"])
    print('查看特征值名字: \n', iris.feature_names)
    print('查看特征值\n', iris.data, iris.data.shape)

    # 数据集的划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print('训练集的特征值：\n', x_train, x_train.shape)

    return None


def dict_demo():
    """
    字典特征提取

    :return:
    """

    data = [{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60}, {'city': '广州', 'temperature': 30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer()

    # 2、调用fit_transferm（）
    data_new = transfer.fit_transform(data)
    print('data_new:\n', data_new)
    print('特征名字\n', transfer.get_feature_names())

    return None


def count_demo():
    """
    文本特征值提取：CountVectorizer

    :return:
    """
    data = ['life is short, I like like python', 'life is too long, I dislike python']
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transfom
    data_new = transfer.fit_transform(data)
    print("data_new\n", data_new.toarray())
    print("特征名字\n", transfer.get_feature_names_out())

    return None


def count_chinese_demo():
    """
    中文 文本特征值提取：CountVectorizer

    :return:
    """
    data = ['我爱 北京', '我爱 北京 天安门, 天安门 上 太阳升']
    # 1、实例化一个转换器类
    transfer = CountVectorizer()
    # 2、调用fit_transfom
    data_new = transfer.fit_transform(data)
    print("data_new\n", data_new.toarray())
    print("特征名字\n", transfer.get_feature_names_out())

    return None


def cut_word(text):
    """
    进行中文分词：“我爱北京天安门”---->“我 爱 北京 天安门”
    :param text:
    :return:
    """
    text = ' '.join(list(jieba.cut(text)))
    return text


def count_chinese_demo2():
    """
    中文文本特征提取，自动分词
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝大部分都是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是再看它的过去。",
            "如果只用一种方式了解某样失误，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new) 打印看下分词效果
    # 1、实例化一个转换器类
    transfer = CountVectorizer(stop_words=['一种', '不会'])
    # 2、调用fit_transfom
    data_final = transfer.fit_transform(data_new)
    print("data_new\n", data_final.toarray())
    print("特征名字\n", transfer.get_feature_names_out())

    return None


def tfidf_demo():
    """
    用TF-IDF的方法，进行文本抽取
    :return:
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝大部分都是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是再看它的过去。",
            "如果只用一种方式了解某样失误，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]

    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new) 打印看下分词效果
    # 1、实例化一个转换器类
    transfer = TfidfVectorizer(stop_words=['一种', '不会'])
    # 2、调用fit_transfom
    data_final = transfer.fit_transform(data_new)
    print("data_new\n", data_final.toarray())
    print("特征名字\n", transfer.get_feature_names_out())

    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv('dating.txt')
    data = data.iloc[:, :3]  # iloc [:行数, :列数]
    print('data:\n', data)

    # 2、实例化一个转换器类
    transfer = MinMaxScaler()

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print('data_new:\n', data_new)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1、获取数据
    data = pd.read_csv('dating.txt')
    data = data.iloc[:, :3]
    print('data:\n', data)
    # 2、实例化一个转换器类
    transfer = StandardScaler()
    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)
    print('data_new:\n', data_new)

    return None

def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1、获取数据
    data = pd.read_csv('')  # 数据文件缺失
    data = data.iloc[:,1:-2] # 1:-2 即显示：从第二列到倒数第三列，负数从后往前数。
    # 2、实例化一个转换器类
    transfer = VarianceThreshold(threshold= 1 ) # (threshold = )设置阈值过滤不重要的特征

    # 3、调用fit_transform
    data_new = transfer.fit_transform(data)

    # 计算两个变量（特征）之间的相关性
    r1 = pearsonr(data['pe_ratio'],data['pd_radio'])
    print('pe_ratio和pd_radio之间的相关性：\n', r1)
    r2 = pearsonr(data['revenue'],data['total_expense'])
    print('revenue和total_expense之间的相关性：\n',r2)

    return None

def pca_demo():
    """
    PCA 降维
    :return:
    """
    # 1、获取数据
    data = [[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    # 2、实例化一个转换器类
    transfer1 = PCA(n_components=0.95)
    transfer2 = PCA(n_components=2)
    # 3、调用fit_transform
    data_new1 = transfer1.fit_transform(data)
    data_new2 = transfer2.fit_transform(data)
    print('保留95%的信息data_new1：\n', data_new1)
    print('降低成为2维data_new2:\n', data_new2)

    return None



if __name__ == "__main__":

    # 代码1：sklearn 数据集的使用
    # datasets_demo()
    # 代码2： 字典特征抽取
    # dict_demo()
    # 代码3： 文本特征抽取：CountVectorizer
    # count_demo()
    # 代码4： 中文文本特征提取
    # count_chinese_demo()
    # 代码5： 中文文本自动分词
    # count_chinese_demo2()
    # 代码6： 中文分词
    # print(cut_word('我爱北京天安门'))
    # 代码7： 用TF-IDF的方法，进行文本抽取
    # tfidf_demo()
    # 代码8： 归一化
    # minmax_demo()
    # 代码9： 标准化
    # stand_demo()
    # 代码10： 过滤低方差特征
    # variance_demo()
    # 代码11： PCA降维
    pca_demo()