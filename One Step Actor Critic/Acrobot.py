# Dymanic comes from the RLbook
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import List
def rk4(derivs, y0, t):
    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float64)
    else:
        yout = np.zeros((len(t), Ny), np.float64)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        this = t[i]
        dt = t[i + 1] - this
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0))
        k2 = np.asarray(derivs(y0 + dt2 * k1))
        k3 = np.asarray(derivs(y0 + dt2 * k2))
        k4 = np.asarray(derivs(y0 + dt * k3))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout[-1][:4]
def wrap(x, m, M):
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x, m, M=None):
    if M is None:
        M = m[1]
        m = m[0]
    # bound x between min (m) and Max (M)
    return min(max(x, m), M)

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
        self.output = nn.Linear(hidden_layer[-1],3) # output layer
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

class Acrobot():
    def __init__(self,hiddenLayerPolicy: List[int], hiddenLayerValue: List[int], AlphaTheta=1e-3,AlphaW=1e-3):
        torch.set_default_dtype(torch.float64)
        self.dt = 0.2
        self.LINK_LENGTH_1 = 1.0
        self.LINK_LENGTH_2 = 1.0
        self.LINK_MASS_1 = 1.0
        self.LINK_MASS_2 = 1.0
        self.LINK_COM_POS_1 = 0.5
        self.LINK_COM_POS_2 = 0.5
        self.LINK_MOI = 1.0
        self.MAX_VEL_1 = 4 * np.pi
        self.MAX_VEL_2 = 9 * np.pi
        self.AVAIL_TORQUE = [1.0,0.0,+1]
        self.torque_noise_max = 0.0
        self.Action = [0,1,2]
        self.hiddenLayerPolicy = hiddenLayerPolicy
        self.hiddenLayerValue = hiddenLayerValue
        self.alphaTheta = AlphaTheta
        self.alphaW = AlphaW
        self.gamma = 0.99
        self.epoch = 1000

    def _get_ob(self,s):
        return torch.tensor(np.array(
            [np.cos(s[0]), np.sin(s[0]), np.cos(s[1]), np.sin(s[1]), s[2], s[3]], dtype=np.float32
        ),dtype=torch.float64)

    def updatePolicy(self, policy, delta, optimizer,At):
        log_policy = torch.log(policy[At])
        loss_2 = -delta * log_policy
        optimizer.zero_grad()
        loss_2.backward(retain_graph=True)
        optimizer.step()

    def updateVF(self, value, G, optimizer):
        loss = F.mse_loss(value, G)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _dsdt(self, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 -np.pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * np.sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2)
            + phi2
        )

        ddtheta2 = (
            a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * np.sin(theta2) - phi2
        ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def _terminal(self,s):
        return bool(-np.cos(s[0]) - np.cos(s[1] + s[0]) > 1.0)
    def step(self,At,s):
        torque = self.AVAIL_TORQUE[At]
        s_augmented = np.append(s,torque)
        ns = rk4(self._dsdt,s_augmented,[0,self.dt])
        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        s = ns
        terminate = self._terminal(s)
        reward = -1 if not terminate else 0
        return torch.tensor(s,dtype=torch.float64),terminate,reward

    def train(self):
        policynn = PolicyNetwork(hidden_layer=self.hiddenLayerPolicy)
        valuenn = ValueNetwork(hidden_layer=self.hiddenLayerValue)
        optimizerV = optim.Adam(valuenn.parameters(), lr=self.alphaW)
        optimizerP = optim.Adam(policynn.parameters(), lr=self.alphaTheta)
        Actions = []
        Durations = []
        for i in range(self.epoch):
            s = torch.tensor(np.random.uniform(low=-0.1, high=0.1, size=(4,)),dtype=torch.float64)
            terminate = False
            step = 0
            actions = []
            while terminate == False and step < 500:
                entries = policynn(s)
                temp = entries.clone()
                At = torch.distributions.Categorical(temp).sample().item()
                actions.append(At)
                snext, terminate, Reward = self.step(At, s)
                state_next = valuenn(snext)
                state_value = valuenn(s)
                delta = Reward + self.gamma * state_next.item() - state_value.item()
                estimate = Reward + self.gamma * state_next
                self.updateVF(value=valuenn(s), G=estimate, optimizer=optimizerV)
                self.updatePolicy(policy=policynn(s), delta=delta, optimizer=optimizerP, At=At)
                step += 1
                s = snext
            print(step)
            Durations.append(step)
            Actions.append(actions)
        return Durations, Actions


if __name__ == '__main__':
    Acrobot = Acrobot(hiddenLayerPolicy=[128], hiddenLayerValue=[128])
    D,_ = Acrobot.train()
    plt.plot([i for i in range(Acrobot.epoch)], D)
    plt.show()



