import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import preprocessing
from RNN1 import RNN

sequence_length = 16  # 序列长度，将图像的每一列作为一个序列
input_size = 11  # 输入数据的维度
hidden_size = 256  # 隐藏层的size
num_layers = 4  # 有多少层

num_classes = 4
batch_size = 128
num_epochs = 20
learning_rate = 0.001


class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction


class RandomAgent(Agent):

    def step(self):
        direction = np.random.randint(0, 4)
        return direction


class ExpectiMaxAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        from .expectimax import board_to_move
        self.search_func = board_to_move

    def step(self):
        direction = self.search_func(self.game.board)
        return direction

class My_Agent_add(Agent):

    def __init__(self, game, display = None):
        self.game = game
        self.display = display
        self.model1 = RNN(11, 256, 4, 4)
        self.model2 = RNN(11, 256, 6, 4)
        self.model3 = RNN(11, 320, 4, 4)
        
        self.model4 = RNN(11, 320, 6, 4)
        self.model5 = RNN(11, 256, 5, 4)
        
        self.model6 = RNN(11, 256, 4, 4)
        self.model7 = RNN(11, 256, 4, 4)
        self.model8 = RNN(11, 256, 5, 4)
        self.model9 = RNN(11, 256, 6, 4)

        self.model10 = RNN(11, 256, 4, 4)
        self.model11 = RNN(11, 256, 4, 4)
        self.model12 = RNN(11, 256, 4, 4)
        self.model13 = RNN(11, 256, 6, 4)


        self.model1.load_state_dict(torch.load('model_RNN_vote1_add_params.pkl'))
        self.model2.load_state_dict(torch.load('model_RNN_vote2_add_params.pkl'))
        self.model3.load_state_dict(torch.load('model_RNN_vote3_add_params.pkl'))
        
        self.model4.load_state_dict(torch.load('model_RNN_vote4_add_params.pkl'))
        self.model5.load_state_dict(torch.load('model_RNN_vote5_add_params.pkl'))
        
        self.model6.load_state_dict(torch.load('model_RNN_vote6_add_params.pkl'))
        self.model7.load_state_dict(torch.load('model_RNN_vote7_add_params.pkl'))
        self.model8.load_state_dict(torch.load('model_RNN_vote8_add_params.pkl'))
        self.model9.load_state_dict(torch.load('model_RNN_vote9_add_params.pkl'))
        
        self.model10.load_state_dict(torch.load('model_RNN_vote10_add_params.pkl'))
        self.model11.load_state_dict(torch.load('model_RNN_vote11_add_params.pkl'))
        
        self.model12.load_state_dict(torch.load('model_RNN_vote12_add_params.pkl'))
        self.model13.load_state_dict(torch.load('model_RNN_vote13_add_params.pkl'))

        self.model1 = self.model1.cuda()
        self.model2 = self.model2.cuda()
        self.model3 = self.model3.cuda()
        
        self.model4 = self.model4.cuda()
        self.model5 = self.model5.cuda()
        
        self.model6 = self.model6.cuda()
        self.model7 = self.model7.cuda()
        self.model8 = self.model8.cuda()
        self.model9 = self.model9.cuda()
        
        self.model10 = self.model10.cuda()
        self.model11 = self.model11.cuda()
        
        self.model12 = self.model12.cuda()
        self.model13 = self.model13.cuda()

        self.enc = preprocessing.OneHotEncoder()
        enc_test = np.load('enc_test.npy')
        self.enc.fit(enc_test)

    def step(self):
        onehot = self.enc.transform((self.game.board % 2048).reshape(-1, 16)).toarray().reshape(-1, 16, 11)
        onehot = torch.Tensor(onehot).cuda()
        #onehot = torch.Tensor(onehot)
        #print(onehot.type())
        direction1 = np.argmax(self.model1(onehot).cpu().detach().numpy())
        direction2 = np.argmax(self.model2(onehot).cpu().detach().numpy())
        direction3 = np.argmax(self.model3(onehot).cpu().detach().numpy())
        
        direction4 = np.argmax(self.model4(onehot).cpu().detach().numpy())
        direction5 = np.argmax(self.model5(onehot).cpu().detach().numpy())
        
        direction6 = np.argmax(self.model6(onehot).cpu().detach().numpy())
        direction7 = np.argmax(self.model7(onehot).cpu().detach().numpy())
        direction8 = np.argmax(self.model8(onehot).cpu().detach().numpy())
        direction9 = np.argmax(self.model9(onehot).cpu().detach().numpy())
        
        direction10 = np.argmax(self.model10(onehot).cpu().detach().numpy())
        direction11 = np.argmax(self.model11(onehot).cpu().detach().numpy())
        
        direction12 = np.argmax(self.model12(onehot).cpu().detach().numpy())
        direction13 = np.argmax(self.model13(onehot).cpu().detach().numpy())

        a = np.zeros((4))

        a[direction1] += 1
        a[direction2] += 1
        a[direction3] += 1
        a[direction4] += 1
        a[direction5] += 1
        a[direction6] += 1
        a[direction7] += 1
        a[direction8] += 1
        a[direction9] += 1
        a[direction10] += 1
        a[direction11] += 1
        a[direction12] += 1
        a[direction13] += 1

        print (a)
        direction = np.argmax(a)    

        return direction
