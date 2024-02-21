import os
import torch
from pyhanlp import *
import numpy as np
from feedbackward_netwark import feedbackward_netwark
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.optim as optim
import pickle
import configparser

# 获取当前脚本文件的上一层目录， 构建路径并创建文件夹
dir_path = os.path.dirname(os.path.abspath(__file__))
words_path = os.path.join(dir_path, "Data" + os.sep + "words.pkl")
train_data_path = os.path.join(dir_path, "Data" + os.sep + "train_data.pkl")
config_path = os.path.join(dir_path, "config.ini")
best_model_path = os.path.join(dir_path, "model.pth")

config = configparser.ConfigParser()
config.read(config_path)
TEST_SIZE  = float(config['DEFAULT']['TEST_SIZE'])
LR = float(config['DEFAULT']['LR'])
EPOCH = int(config['DEFAULT']['EPOCH'])

def read_pkl(path):
    with open(path, 'rb') as f_words:
        return pickle.load(f_words)

def load_data():
    words = read_pkl(words_path)
    train_data = np.array(read_pkl(train_data_path))

    # 划分训练集和测试集
    X = list(train_data[:, 0])
    y = list(train_data[:, 1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
    # 转为Tensor
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train)
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test)
    X = torch.tensor(X).float()
    y = torch.tensor(y)

    return X_train, y_train, X_test, y_test, X, y, words

def train_model(model, X_train, y_train, X_test, y_test, optimizer, criterion):
    # 记录训练和测试损失、准确率
    train_arr = {'loss': [], 'acc': []}
    test_arr = {'loss': [], 'acc': []}

    for epoch in range(EPOCH):
        # 训练阶段
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()
        optimizer.step()

        # 记录训练损失和准确率
        train_loss = loss.item()
        _, pred = out.max(1)
        train_acc = (pred == y_train).sum().item() / len(y_train)
        train_arr['loss'].append(train_loss)
        train_arr['acc'].append(train_acc)

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print("Train Epoch [{}/{}] Loss: {:.3f} Acc: {:.3f}".format(epoch + 1, EPOCH, train_loss, train_acc))

        # 测试阶段
        model.eval()
        with torch.no_grad():
            out = model(X_test)
            loss = criterion(out, y_test)

            # 记录测试损失和准确率
            test_loss = loss.item()
            _, pred = out.max(1)
            test_acc = (pred == y_test).sum().item() / len(y_test)
            test_arr['loss'].append(test_loss)
            test_arr['acc'].append(test_acc)

            # 打印测试信息
            if (epoch + 1) % 10 == 0:
                print("Test Epoch [{}/{}] Loss: {:.3f} Acc: {:.3f}".format(epoch + 1, EPOCH, test_loss, test_acc))

    return train_arr, test_arr

def save_plots(train_arr, test_arr):
    # 绘制训练过程中的损失和准确率曲线
    for phase in ['loss', 'acc']:
        plt.figure()
        plt.grid()
        plt.title(phase)
        plt.plot(train_arr[phase], label='Train ' + phase)
        plt.plot(test_arr[phase], label='Test ' + phase)
        plt.legend()

        # 保存图片
        img_path = os.path.join(os.getcwd(), 'img' + os.sep + phase + '.jpg')
        plt.savefig(img_path)

def train_final_model(model, X, y, optimizer, criterion):
    # 初始化最佳模型的损失和正确率
    best_loss = float('inf')  # 初始设置为正无穷大
    best_acc = 0.0
    best_model_params = None

    # 使用全部数据进行训练，得到最终模型
    print("==========使用全部数据进行训练，得到最终模型==========")
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        # 记录损失和正确数
        final_loss = loss.item()
        _, pred = out.max(1)
        final_acc = (pred == y).sum().item() / len(y)

        # 打印训练信息
        if (epoch + 1) % 10 == 0:
            print("Final Epoch [{}/{}] Loss: {:.3f} Acc: {:.3f}".format(epoch + 1, 300, final_loss, final_acc))

        # 更新最佳模型参数
        if final_acc >= best_acc:
            best_acc = final_acc
            best_loss = final_loss
            best_model_params = model.state_dict()

    # 打印选择的最佳模型的信息
    print("Best Model: Loss: {:.3f} Acc: {:.3f}".format(best_loss, best_acc))
    torch.save(best_model_params, best_model_path)

if __name__ == "__main__":
    # 加载数据
    X_train, y_train, X_test, y_test, X, y, words = load_data()

    # 模型、优化器和损失函数
    model = feedbackward_netwark(len(words), 15)# 初始化模型，一共15种类别
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    train_arr, test_arr = train_model(model, X_train, y_train, X_test, y_test, optimizer, criterion)

    # 保存训练过程中的曲线图
    save_plots(train_arr, test_arr)

    # 训练最终模型
    best_model_params = train_final_model(model, X, y, optimizer, criterion)
