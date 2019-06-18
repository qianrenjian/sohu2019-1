import jieba
import re
import torch
import torch.nn as nn

count = 0
len_title = 0
max_title = 0
max_text = 0
len_text = 0
with open(r"/home/zxl/Projects/sohu/data/coreEntityEmotion_train.txt", 'r') as f:
    for line in f:
        label_list = []
        # 找到label那一部分字符串
        label_part = re.findall(r'"coreEntityEmotions": (.*)}], "title":', line)
        # 去除符号
        cleaned_label_part = re.findall(r'''([^",:{}\[\]']*)''', str(label_part))
        for word in cleaned_label_part:
            if word != '' and word != ' ':
                label_list.append(word)

        if len(label_list) != 0:
            emotion = []
            aspect = []
            for i in range(int(len(label_list) / 4)):
                emotion.append(label_list[4 * i + 3])
                aspect.append(label_list[4 * i + 1])
        else:
            emotion = None
            aspect = None

        newsId = re.findall(r'{"newsId": "(.*)", "coreEntityEmotions"', line)[0]
        title = re.findall(r', "title": "(.*)", "content": "', line)[0]
        text = re.findall(r'", "content": "(.*)"}', line)[0]
        title = jieba.lcut(title, cut_all=False)
        print(aspect)
        print(jieba.lcut(aspect[0], cut_all=False))
        input()
        # if len(title) > max_title:
        #     max_title = len(title)
        # if len(text) > max_text:
        #     max_text = len(text)
        # len_title += len(title)
        # len_text += len(text)
        # count += 1
        # print(newsId, aspect, emotion, title)
        # print(text)
        # input()
        # print(count)

# print("title: %4d  max: %4d | text: %4d  max: %4d" % (len_title / count, max_title, len_text / count, max_text))
# title:   13  max:   42 | text: 1315  max: 18532

