import torch
import os
import re
import jieba
from torchtext import data
__all__ = torch


def sohu_data_reader(path, train=True, dev=False, test=False, toy=False,
                     extraction=True, classification=False,
                     text_field=None, title_field=None, aspect_filed=None, emotion_field=None):
    """
    搜狐2019校园算法大赛 ASBA数据集处理
    Args:
         path: data文件夹路径
         train: 训练集
         dev: 测试集
         test: 提交
         toy: 用example测试是否正常运行
         extraction: aspect提取任务
         classification: aspect情感分类任务
         text_field: 文章field
         title_field: 标题field
         aspect_filed: aspect词field
    """
    # 选择数据路径
    if toy:
        path = os.path.join(path, "coreEntityEmotion_example.txt")
    elif train or dev:
        path = os.path.join(path, "coreEntityEmotion_train.txt")
    else:
        path = os.path.join(path, "coreEntityEmotion_test_stage1.txt")

    with open(path, "r") as f:

        examples = []
        fields = [("text", text_field), ("title", title_field), ("aspect", aspect_filed)]

        for line in f:
            label_list = []
            # 找到label那一部分字符串
            if toy:
                label_part = re.findall(r'"coreEntityEmotions": (.*)}], "content":', line)
            else:
                label_part = re.findall(r'"coreEntityEmotions": (.*)}], "title":', line)
            # 去除符号
            cleaned_label_part = re.findall(r'''([^",:{}\[\]']*)''', str(label_part))
            for word in cleaned_label_part:
                if word != '' and word != ' ':
                    label_list.append(word)

            if not test:
                if extraction:
                    aspect = []
                    for i in range(int(len(label_list) / 4)):
                        aspect.append(label_list[4 * i + 1])
                elif classification:
                    emotion = []
                    for i in range(int(len(label_list) / 4)):
                        emotion.append(label_list[4 * i + 3])

            # example的提取格式不一样
            if toy:
                newsId = re.findall(r'{"newsId": "(.*)", "title"', line)[0]
                title = re.findall(r', "title": "(.*)", "coreEntityEmotions":', line)[0]
                text = re.findall(r'"content": "(.*)"}', line)[0]
                title = jieba.lcut(title, cut_all=False)
                text = jieba.lcut(text, cut_all=False)
            else:
                newsId = re.findall(r'{"newsId": "(.*)", "coreEntityEmotions"', line)[0]
                title = re.findall(r', "title": "(.*)", "content": "', line)[0]
                text = re.findall(r'", "content": "(.*)"}', line)[0]
                title = jieba.lcut(title, cut_all=False)
                text = jieba.lcut(text, cut_all=False)

            while ' ' in title:
                title.remove(' ')
            while ' ' in text:
                text.remove(' ')

            examples.append(data.Example.fromlist([text, title, aspect], fields))
        examples = data.Dataset(examples, fields)
        return examples
