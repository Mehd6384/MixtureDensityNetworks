import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.distributions import Normal 

import numpy as np 
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

class MDN(nn.Module): 

	def __init__(self, h = 64, k = 5): 


		# H corresponds to the number of hidden neurons
		# k is the number of gaussians 

		nn.Module.__init__(self)

		self.l1 = nn.Linear(1,h)

		self.pis = nn.Linear(h,k)
		self.mus = nn.Linear(h,k)
		self.stds= nn.Linear(h,k)

	def forward(self,x): 

		z = F.tanh(self.l1(x))

		pis = F.softmax(self.pis(z), dim = 1)
		mus = self.mus(z)
		stds = F.softplus(self.stds(z))

		return pis, mus, stds 


def get_prob(y_real, pis, mus, stds): 

	dists = Normal(mus, stds)  # Generating the gaussians 
	probs = torch.exp(dists.log_prob(y_real)) # What is the probability of observations given the gaussians 
	vals = torch.sum(probs*pis, dim = 1) # Adding the distributions according to the parameters pi 
	losses = -torch.log(vals) # Final loss 

	return torch.mean(losses)


mdn = MDN() #Model 
adam = optim.Adam(mdn.parameters()) # Opt 

def update(l): # Run one optimization 
	adam.zero_grad()
	l.backward()
	adam.step()

# Creating the dataset -----------

x = torch.linspace(-10.,10,1000).reshape(-1,1)
noise = torch.randn(x.shape[0],1)
y = 0.5*x + 7*torch.sin(x*0.75) + noise 


# Setting up training loop  -----------


epochs = 20000
batch_size = 64

# Loop ----------------------
mean_loss = 0. 
for epoch in range(1,epochs+1): 

	ind = np.random.randint(0,x.shape[0]-(batch_size+1)) # sample from batch 

	pis, means, stds = mdn(x[ind:ind+batch_size,:]) #model prediction 
	loss = get_prob(y[ind:ind+batch_size,:], pis, means, stds) # loss 
	update(loss) 

	mean_loss += loss.item()
	if epoch%500 == 0:
		print('Epoch {} - Loss {:.8f}'.format(epoch, mean_loss/500.))
		mean_loss = 0.

x_test = torch.linspace(-10, 10, 1000).view(-1, 1)

pis, means, stds = mdn(x_test)
selection = torch.max(pis, 1)[1]

k = torch.multinomial(pis, 1).view(-1)
y_pred = torch.normal(means, stds)[np.arange(1000), k].detach().numpy()
points = torch.gather(means, dim = 1, index = selection.reshape(-1,1)).detach().numpy()
# input(y_pred.shape)

# input(points)
points = torch.sum(pis*means, dim = 1).detach().numpy()

plt.scatter(y,x, label = 'real')
plt.scatter(points, x_test, label = 'pred')
plt.scatter(y_pred, x_test, label = 'pred')


plt.legend()
plt.show()
