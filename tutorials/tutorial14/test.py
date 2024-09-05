from pina.model.layers import OrthogonalBlock
from pina import LabelTensor
import torch

torch.manual_seed(11)

orth = OrthogonalBlock(0)

x = torch.randn(3,5,requires_grad=True)
x = LabelTensor(x,[f'u{i}' for i in range(x.size(1))])

y = orth(x.tensor)

print(x)
print("---------------")
print(y)
print("---------------")
print(torch.matmul(y,y.T))

print(y.requires_grad)

z = torch.mean(y)
z.backward()
print(z.grad)
