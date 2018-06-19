import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Normal 
from mdn_layer import MDNLayer

import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')


class P_Model(nn.Module): 

	def __init__(self): 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(1,64)
		self.l2 = MDNLayer(64,5) #64 hidden, 5 gaussians 

	def forward(self, x): 

		x = F.relu(self.l1(x))
		pis, means, stds = self.l2(x)

		return pis, means, stds 

	def compute_loss(self, x, y): 

		x = F.relu(self.l1(x))
		loss = self.l2.get_loss(x,y)
		return loss 

	def sample(self, x): 

		x = F.relu(self.l1(x))
		results = self.l2.sample(x)
		return results

	def sample_max(self, x): 

		x = F.relu(self.l1(x))
		results = self.l2.sample_max(x)
		return results
	def sample_categorical(self, x): 

		x = F.relu(self.l1(x))
		results = self.l2.sample_categorical(x)
		return results

n_samples = 1000

epsilon = torch.randn(n_samples)
x_data = torch.linspace(-10, 10, n_samples)
y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

y_data, x_data = x_data.view(-1, 1), y_data.view(-1, 1)
x_test = torch.linspace(-15, 15, n_samples).view(-1, 1)

model = P_Model()
batch_size = 64

adam = optim.Adam(model.parameters(), 5e-3)

def update(l): 
	adam.zero_grad()
	l.backward()
	adam.step()

mean_loss = 0. 
import time 
for epoch in range(1,20000): 

	ind = np.random.randint(0,n_samples - (batch_size+1), (batch_size,1))
	xt = torch.gather(x_data, dim = 0, index = torch.tensor(ind).long())
	yt = torch.gather(y_data, dim = 0, index = torch.tensor(ind).long())

	start = time.time()
	loss = model.compute_loss(xt,yt)
	end = time.time()

	input(end - start)
	update(loss)

	mean_loss += loss.item()
	if(epoch % 100 == 0): 
		print('{:.6f}'.format(mean_loss/100.))	
		mean_loss = 0. 

		points = model.sample_max(x_data).detach().numpy() 
		plt.cla()
		plt.scatter(x_data.numpy(), y_data.numpy())
		plt.scatter(x_data.numpy(), points)
		plt.ylim(-15,15)
		plt.pause(0.1)

