import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
N第二周作业，自己设计多分类任务完成模型训练，如果不知道设计什么任务，就做以下任务:
对于任意五维向量，它最大数字在第几维就属于第几类
"""

class TorchModel(nn.Module):
    def __init__(self,input_size,output_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,output_size)     #线性层
        self.loss = nn.functional.cross_entropy     #loss函数采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self,x,y=None):
        logits = self.linear(x)      # (batch_size, input_size) -> (batch_size, output_size) 
        if y is not None:
            return self.loss(logits,y)      #logits和真实值计算损失`
        else:
            return logits      #输出预测结果

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，最大的一个值所在的维度就是它的类别             
def build_sample():
    x = np.random.random(5) 
    max_index = np.argmax(x)    #最大值所在维度 = 真实标签 0~4
    y = max_index   
    return x, y

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()  #设置模式为评估模式
    test_sample_num = 100
    x_test,y_test = build_dataset(test_sample_num)  #真实结果
    correct_num, wrong_num = 0, 0
    with torch.no_grad():  #不计算梯度
        logits = model(x_test)  #预测原始打分
        y_pred = torch.argmax(logits, dim=1)  #取最大值所在维度作为预测类别
        for y_pred,y_test in zip(y_pred,y_test):
            if y_pred == y_test:
                correct_num += 1
            else:
                wrong_num += 1
    print("正确预测: %d, 错误预测: %d" % (correct_num, wrong_num))
    return correct_num / (correct_num + wrong_num)

#主函数
def main():
    #配置参数
    epoch_num = 1000    #训练轮数
    batch_size = 50     #每次训练样本数
    total_sample_num = 1000    #总样本数
    learning_rate = 0.01    #学习率
    input_size = 5      #输入维度
    output_size = 5     #输出维度
    #建立模型  
    model = TorchModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []       
    # 创建训练集，正常任务是读取训练集
    train_x,train_y = build_dataset(total_sample_num)  #生成训练数据集
    for epoch in range(epoch_num):
        model.train()  #设置模式为训练模式
        watch_loss = []        
        for batch_index in range(total_sample_num // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(x, y)
            optim.zero_grad()  #清空梯度
            loss.backward()  #反向传播计算梯度
            optim.step()  #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))

        # 测试本轮模型结果  
        acc = evaluate(model)
        log.append([acc,float(np.mean(watch_loss))])
    #保存模型
    torch.save(model.state_dict(), "torch_model.bin")
    #画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    main()


