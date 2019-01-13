from game2048.game import Game
from game2048.displays import Display
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn import preprocessing

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display=Display(), **kwargs)
    agent.play(verbose=True)
    return game.score


if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 50
    N_I = 10

    '''====================
    Use your own agent here.'''
    #model = torch.load('model_669.pkl')
    from game2048.agents import My_Agent_add as TestAgent
    '''===================='''
    #while (1):
    #scores_all = []
    #for nn in range(N_I):
    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                                 AgentClass=TestAgent)
        scores.append(score)
        #socres_all.append(sum(scores) / len(scores))
    #        if (score < 2048):
    #            break
    #    if (sum(scores) / len(scores) >= 2000): 
    #        break

    #print (scores)
    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
