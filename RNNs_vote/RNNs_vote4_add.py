from game2048.game import Game
from game2048.displays import Display, IPythonDisplay
from game2048.agents import Agent, RandomAgent, ExpectiMaxAgent
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm
import time

#from sklearn import preprocessing

BATCH_SIZE = 128

display1 = Display()
display2 = IPythonDisplay()

'''
train_x_512 = np.load('data_x_final_enc_512.npy').reshape(-1, 16)
train_y_512 = np.load('data_y_final_512.npy')
'''
train_x1 = np.load("/cluster/home/it_stu91/2048/data_x_final_enc_0_256.npy")
train_y1 = np.load("/cluster/home/it_stu91/2048/data_y_final_0_256.npy")

train_x2 = np.load("/cluster/home/it_stu91/2048/data_x_final_enc_256_512.npy")
train_y2 = np.load("/cluster/home/it_stu91/2048/data_y_final_256_512.npy")

train_x3 = np.load("/cluster/home/it_stu91/2048/data_x_final_enc_1024.npy")
train_y3 = np.load("/cluster/home/it_stu91/2048/data_y_final_1024.npy")

train_x4 = np.load("/cluster/home/it_stu91/2048/data_x_final_enc_2048.npy")
train_y4 = np.load("/cluster/home/it_stu91/2048/data_y_final_2048.npy")

train_x5 = np.load("/cluster/home/it_stu91/2048/data_x_add_enc.npy")
train_y5 = np.load("/cluster/home/it_stu91/2048/data_y_add.npy")

train_x6 = np.load("/cluster/home/it_stu91/2048/data_x_add2_enc.npy")
train_y6 = np.load("/cluster/home/it_stu91/2048/data_y_add2.npy")
'''
train_x_2048 = np.load('data_x_final_enc_2048.npy').reshape(-1, 16)
train_y_2048 = np.load('data_y_final_2048.npy')
'''
'''
train_x_512 = torch.Tensor(train_x_512)
train_y_512 = torch.LongTensor(train_y_512)
'''
train_x = np.zeros((0, 16, 11))
train_y = np.zeros((0, 1))

train_x = np.vstack((train_x, train_x1))
train_x = np.vstack((train_x, train_x2))
train_x = np.vstack((train_x, train_x3))
train_x = np.vstack((train_x, train_x4))
train_x = np.vstack((train_x, train_x5))
train_x = np.vstack((train_x, train_x6))


train_y = np.vstack((train_y, train_y1))
train_y = np.vstack((train_y, train_y2))
train_y = np.vstack((train_y, train_y3))
train_y = np.vstack((train_y, train_y4))
train_y = np.vstack((train_y, train_y5))
train_y = np.vstack((train_y, train_y6))
'''
train_x_2048 = np.load('data_x_final_enc_2048.npy').reshape(-1, 16)
train_y_2048 = np.load('data_y_final_2048.npy')
'''
'''
train_x_512 = torch.Tensor(train_x_512)
train_y_512 = torch.LongTensor(train_y_512)
'''
train_x = torch.Tensor(train_x)
train_y = torch.LongTensor(train_y)
'''
train_x_2048 = torch.Tensor(train_x_2048)
train_y_2048 = torch.LongTensor(train_y_2048)
'''

class MyDataset(data.Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):#返回的是tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)
'''
mydataset_512 = MyDataset(train_x_512, train_y_512)
train_loader_512 = data.DataLoader(mydataset_512, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
'''
mydataset = MyDataset(train_x, train_y)
train_loader = data.DataLoader(mydataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
'''
mydataset_2048 = MyDataset(train_x_2048, train_y_2048)
train_loader_2048 = data.DataLoader(mydataset_2048, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
'''




sequence_length = 16  # 序列长度，将图像的每一列作为一个序列
input_size = 11  # 输入数据的维度
hidden_size = 320  # 隐藏层的size
num_layers = 6  # 有多少层

num_classes = 4
batch_size = 128
num_epochs = 20
learning_rate = 0.001


'''
class RNN_512(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_512, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        
        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out

model_512 = RNN_512(input_size, hidden_size, num_layers, num_classes)
model_512 = model_512.cuda()
'''

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        
        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes)
model = model.cuda()

'''
class RNN_2048(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_2048, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda()) 
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda())
        
        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)
        
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out

model_2048 = RNN_1024(input_size, hidden_size, num_layers, num_classes)
model_2048 = model_2048.cuda()

'''
criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
#optimizer_512 = torch.optim.Adam(model_512.parameters(), lr = 0.001)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#optimizer_2048 = torch.optim.Adam(model_2048.parameters(), lr = 0.001)


model = model.cuda()
NUM_EPOCHS = 8
for epoch in range(NUM_EPOCHS):
    i = 0
    running_loss = 0
    print('EPOCHS', epoch + 1)
    correct = 0
    for images, labels in tqdm(train_loader):
        i += 1
        images, labels = Variable(images), Variable(labels)
        #print(images.shape)
        labels = labels.long()
        optimizer.zero_grad()
        #print(images.shape)
        images = images.reshape(-1, 16, 11).cuda()
        output = model(images).reshape(-1, 4).cuda()
        labels = labels.float().reshape(-1).cuda()
        correct += (labels.cpu().numpy() == output.cpu().detach().numpy().argmax(axis = 1)).sum()
        #print(output.shape)
        #print(labels.shape)
        loss = criterion(output, labels.long())
        running_loss += float(loss)
        loss.backward()
        optimizer.step()
    print(running_loss/i)
    print("accuracy: ", correct/float(train_x.shape[0]))
    #print("accuracy: ", correct/float(train_x.shape[0]))
#torch.save(model_1024, '/cluster/home/it_stu84/2048/model_1024.pkl')
torch.save(model.state_dict(), '/cluster/home/it_stu91/2048/model_RNN_vote4_add_params.pkl')