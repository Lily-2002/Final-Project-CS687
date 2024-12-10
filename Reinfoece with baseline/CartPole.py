import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
import random
class PolicyNetwork(nn.Module):
    def __init__(self,hidden_layer = [128]):
        super().__init__()
        Layers = []
        for i in range(len(hidden_layer)):
            if i == 0:
                Layers.append(nn.Linear(4,hidden_layer[i]))
            else:
                Layers.append(nn.Linear(hidden_layer[i - 1],hidden_layer[i]))
            # Layers.append(nn.ReLU())
        self.output = nn.Linear(hidden_layer[-1],2) # output layer
        self.layers = nn.Sequential(
            *Layers,
            nn.ReLU()
        )
    def forward(self,x):
        x = self.layers(x)
        # x = F.relu(x)
        actions = self.output(x)
        probs = F.softmax(actions,dim=-1)
        return probs
class ValueNetwork(nn.Module):
    def __init__(self, hidden_layer=[128]):
        super().__init__()
        Layers = []
        for i in range(len(hidden_layer)):
            if i == 0:
                Layers.append(nn.Linear(4, hidden_layer[i]))
            else:
                Layers.append(nn.Linear(hidden_layer[i - 1], hidden_layer[i]))
        self.output = nn.Linear(hidden_layer[-1], 1)  # output layer
        self.layers = nn.Sequential(
            *Layers,
            nn.ReLU()
        )

    def forward(self, x):
        x = self.layers(x)
        # x = F.relu(x)
        state = self.output(x)
        return state

class CartPole():
    def __init__(self, hiddenLayerPolicy: List[int], hiddenLayerValue: List[int], AlphaTheta=1e-4, AlphaW=1e-3):
        self.G = 9.8
        self.MC = 1.0  # Mcart
        self.MP = 0.1  # Mport
        self.length = 0.5  # length of pole
        self.fix_force = 10
        self.Limit_theta = math.pi / 15
        self.Limit_X = 2.4
        self.Action = [0, 1]
        self.M = self.MP + self.MC
        self.DeltaT = 0.02
        torch.set_default_dtype(torch.float64)
        # self.state = torch.tensor([0, 0, 0, 0], dtype=torch.float64)  # x,v,theta,Dtheta
        self.hiddenLayerPolicy = hiddenLayerPolicy
        self.hiddenLayerValue = hiddenLayerValue
        self.alphaTheta = AlphaTheta
        self.alphaW = AlphaW
        self.gamma = 0.99
        self.epoch = 1000

    def step(self, At,state):
        x, dot_x, theta, dot_theta = state
        F = self.fix_force if At == 1 else -self.fix_force
        # Using Barto et.al to simulate (without considering the friction between the cart and track)
        t = (-F - self.MP * self.length * dot_theta * math.sin(theta)) / (self.MP + self.MC)
        ddotTheta = (self.G * math.sin(theta) + math.cos(theta) * t) / (self.length * (4 / 3 - (self.MP * math.cos(theta) ** 2) / (self.MC + self.MP)))
        ddotX = -t - ddotTheta * math.cos(theta) / (self.MP + self.MC)
        x = x + self.DeltaT * dot_x
        dot_x = dot_x + self.DeltaT * ddotX
        theta = theta + self.DeltaT * dot_theta
        dot_theta = dot_theta + self.DeltaT * ddotTheta
        state = torch.tensor([x, dot_x, theta, dot_theta], dtype=torch.float64)
        terminate = bool(abs(x) > self.Limit_X or abs(theta) > self.Limit_theta)
        Reward = 1 if not terminate else 0
        return state, terminate, Reward

    def GenerateEpisode(self, nn):
        Policies = []
        States = []
        Rewards = []
        s = torch.tensor([0, 0, 0, 0], dtype=torch.float64)
        terminate = False
        step = 0
        while not terminate and step < 500:
            States.append(s)
            entries = nn(s)
            temp = entries.clone()
            At = torch.distributions.Categorical(temp).sample().item()
            Policies.append(At)
            s, terminate, Reward = self.step(At, s)
            Rewards.append(Reward)
            step += 1
        return States, Policies, Rewards, step

    def updatePolicy(self, policy, delta, optimizer,At):
        log_policy = torch.log(policy[At])
        loss_2 = -delta * log_policy
        optimizer.zero_grad()
        loss_2.backward()
        optimizer.step()

    def updateVF(self, value, G, optimizer):
        G = torch.tensor(G, dtype=torch.float64,requires_grad=False).unsqueeze(0)
        loss = F.mse_loss(value, G)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self):
        torch.autograd.set_detect_anomaly(True)
        policynn = PolicyNetwork(hidden_layer=self.hiddenLayerPolicy)
        valuenn = ValueNetwork(hidden_layer=self.hiddenLayerValue)
        Durations = []
        optimizerV = optim.AdamW(valuenn.parameters(), lr=self.alphaW,weight_decay=5e-3)
        optimizerP = optim.AdamW(policynn.parameters(), lr=self.alphaTheta,weight_decay=5e-3)
        Actions = []
        for i in range(self.epoch):
            S, P, R, Duration = self.GenerateEpisode(nn=policynn)
            print(Duration)
            Actions.append(P)
            Durations.append(Duration)
            for t in range(len(S)):
                G = 0
                for k in range(t + 1, len(S)):
                    G = G + self.gamma ** (k - t - 1) * R[k]
                value = valuenn(S[t]).detach()
                delta = G - value
                self.updateVF(value=valuenn(S[t]), G=G, optimizer=optimizerV)
                self.updatePolicy(policy=policynn(S[t]), delta=delta, optimizer=optimizerP,At=P[t])


        torch.save(policynn.state_dict(), r"C:\Users\fulian\Desktop\Final_project\Reinfoece with baseline\Cart_Pole_policy_net.pth")
        torch.save(valuenn.state_dict(), r"C:\Users\fulian\Desktop\Final_project\Reinfoece with baseline\Cart_Pole_value_network.pth")
        return Durations,Actions
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pickle

    CartPole = CartPole(hiddenLayerPolicy=[128], hiddenLayerValue=[128])
    D, A = CartPole.train()
    with open(r"C:\Users\fulian\Desktop\Final_project\Reinfoece with baseline\A_Cart.pkl", 'wb') as f:
        pickle.dump(A, f)
    with open(r"C:\Users\fulian\Desktop\Final_project\Reinfoece with baseline\A_Cart.pkl", 'rb') as f:  # 'rb'表示以二进制模式读取文件
        loaded_list = pickle.load(f)
    plt.plot([i for i in range(CartPole.epoch)], D)
    plt.show()











