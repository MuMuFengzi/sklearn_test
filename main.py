import sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import jieba




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

    data = [{'city':'北京','temperature':100},{'city':'上海','temperature':60},{'city':'广州','temperature':30}]
    # 1、实例化一个转换器类
    transfer = DictVectorizer()

    #2、调用fit_transferm（）
    data_new = transfer.fit_transform(data)
    print('data_new:\n', data_new)
    print('特征名字\n', transfer.get_feature_names())

    return None

def count_demo():
    """
    文本特征值提取：CountVectorizer

    :return:
    """
    data = ['life is short, I like like python','life is too long, I dislike python']
    #1、实例化一个转换器类
    transfer = CountVectorizer()
    #2、调用fit_transfom
    data_new = transfer.fit_transform(data)
    print("data_new\n", data_new.toarray())
    print("特征名字\n", transfer.get_feature_names_out())

    return None

def count_chinese_demo():
    """
    中文 文本特征值提取：CountVectorizer

    :return:
    """
    data = ['我爱 北京','我爱 北京 天安门, 天安门 上 太阳升']
    #1、实例化一个转换器类
    transfer = CountVectorizer()
    #2、调用fit_transfom
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
    print(data_new)
    #1、实例化一个转换器类


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
    count_chinese_demo2()
