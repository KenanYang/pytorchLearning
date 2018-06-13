import torch
import numpy as np
from torch.autograd import Variable

# data = [[1,2],[3,4]]
# tensor = torch.FloatTensor(data) #32-bit floating point
# data  = np.array(data)

# print(
#   '\nnumpy: ', np.matmul(data,data),
#   '\ntorch: ', torch.mm(tensor,tensor)
#   )

# print(
#   '\nnumpy: ', data.dot(data),
#   '\ntorch: ', tensor.dot(tensor)
#   )

# -- variables --
tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor * tensor)
v_out = torch.mean(variable * variable)


# print(t_out)
# print(v_out)

v_out.backward()

print(variable.grad)

print(variable)

print(variable.data)

print(variable.data.numpy())
