import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 64
TIME_STEP = 28      # rnn 时间步数 / 图片高度
INPUT_SIZE = 28     # rnn 每步输入值 / 图片每行像素
LR = 0.01           # learning rate
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 Fasle

# Mnist 手写数字
train_data = dsets.MNIST(
    root='/Users/kenanyang/Desktop/mnist/',    # 保存或者提取位置
    train=True,  # this is training data
    transform=transforms.ToTensor(),    # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,          # 没下载就下载, 下载了就不用再下了
)

test_data = dsets.MNIST(root='/Users/kenanyang/Desktop/mnist/', train=False)
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# 为了节约时间, 我们测试时只测试前2000个
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255.   # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:2000]


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        r_out, (h_n, h_c) = self.rnn(x, None)
        # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])  # choose the last time step as output
        return out


rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted

# training and testing
for epoch in range(EPOCH):
    for step, (x, b_y) in enumerate(train_loader):   # gives batch data
        b_x = x.view(-1, 28, 28)   # reshape x to (batch, time_step, input_size)

        output = rnn(b_x)               # rnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 50 == 0:
            test_output = rnn(test_x.view(-1, 28, 28))
            pred_y = torch.max(test_output, 1)[1]  # .squeeze()
            # print(pred_y)
            # print(test_y.size(0))
            accuracy = float(sum(pred_y.numpy() == test_y.numpy())) / float(test_y.size(0))

            print('Epoch: ', epoch, '| train loss: %.4f' % loss[0], '| test accuracy: %.4f' % accuracy)

test_output = rnn(test_x[:10].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')
