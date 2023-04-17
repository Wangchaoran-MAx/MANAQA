import torch
import torch.nn.functional as F
import numpy as np

input = torch.randn(1, 3,50, 22)
x = input.view(input.size(0), -1)
fc=torch.nn.Linear(224*224,224*224)
x=fc(x)
x = x.view(x.size(0), -1, 224, 224)
print(np.shape(x))