import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.distributions import Normal, Categorical 


class MDNLayer(nn.Module): 
	
	def __init__(self, h, k): 

		nn.Module.__init__(self)
		
		self.pis = nn.Linear(h,k)
		self.means = nn.Linear(h,k)
		self.stds = nn.Linear(h,k)

	def forward(self, x): 

		pis = F.softmax(self.pis(x), dim = 1)
		means = self.means(x)
		stds = F.softplus(self.stds(x))

		return pis, means, stds 

	def get_loss(self,x,y): 

		pis, means, stds = self(x)
		dists = Normal(means, stds)
		probs = torch.exp(dists.log_prob(y))
		vals  = torch.sum(probs*pis, dim = 1)
		return -torch.mean(torch.log(vals))

	def sample(self, x): 

		pis, means, stds = self(x)
		
		k = torch.multinomial(pis, 1).view(-1)
		points = torch.normal(means, stds)[torch.arange(x.shape[0]).long(), k]

		points = torch.sum(pis*means, dim = 1)

		return points

	def sample_categorical(self, x): 
		pis, means, stds = self(x)
		categorical = Categorical(pis).sample().reshape(-1,1)
		points = torch.gather(means, dim = 1, index = categorical)

		return points 

	def sample_max(self, x): 
		pis, means, stds = self(x)
		pis_selection = torch.max(pis, dim = 1)[1]
		points = torch.gather(means, dim = 1, index = pis_selection.reshape(-1,1))

		return points 