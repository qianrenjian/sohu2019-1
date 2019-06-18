import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
import torchtext.vocab as vocab
from cnn import CNN
from sohu_data_reader import sohu_data_reader

BATCH_SIZE = 1
LEARNING_RATE = 0.01
EMBEDDING_DIM = 200
CONV_NUM = 4
DROPOUT = 0.55
EPOCH = 40
device = torch.device("cpu")  # GPU


def main():
    # Field
    TEXT = data.Field(batch_first=True)
    TITLE = data.Field(batch_first=True)
    ASPECT = data.Field(batch_first=True)

    # Load data
    train = sohu_data_reader("./data", train=False, dev=False, test=False, toy=True,
                             extraction=True, classification=False,
                             text_field=TEXT, title_field=TITLE,
                             aspect_filed=ASPECT, emotion_field=None)
    dev = sohu_data_reader("./data", train=False, dev=False, test=False, toy=True,
                           extraction=True, classification=False,
                           text_field=TEXT, title_field=TITLE, aspect_filed=ASPECT, emotion_field=None)

    # 建立词表
    TEXT.build_vocab(train)
    TITLE.build_vocab(train)
    ASPECT.build_vocab(train)
    # NEWSID.build_vocab(train)

    # 迭代器
    train_iter, dev_iter = data.BucketIterator.splits((train, dev), batch_size=BATCH_SIZE, shuffle=False,
                                                      repeat=False, device=device, sort=False,
                                                      sort_key=lambda x: len(x.text))

    net = CNN(text=TEXT, emb_dim=EMBEDDING_DIM, conv_num=CONV_NUM, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCH):

        for i, data_iter in enumerate(train_iter):
            text_iter = data_iter.text
            title_iter = data_iter.title
            aspect_iter = data_iter.aspect

            c = 0
            for b in text_iter[0]:
                if aspect_iter[0][0] == b:
                    c = 1
            print(c)
            # aspect_pred = net(train_iter, text_iter)


if __name__ == '__main__':
    main()
