import torch

a = torch.tensor([[[[[1.1, 1.2],[1.3,1.4]],[[1.5, 1.6],[1.7,1.8]]],[[[1.1, 1.2],[1.3,1.4]],[[1.5, 1.6],[1.7,1.8]]]],[[[[2.1, 2.2],[2.3,2.4]],[[2.5, 2.6],[2.7,2.8]]],[[[2.1, 2.2],[2.3,2.4]],[[2.5, 2.6],[2.7,2.8]]]]])
a1 = torch.tensor([[[[1.1, 1.2],[1.3,1.4]],[[1.5, 1.6],[1.7,1.8]]],[[[1.1, 1.2],[1.3,1.4]],[[1.5, 1.6],[1.7,1.8]]]])
a2 = torch.tensor([[[[2.1, 2.2],[2.3,2.4]],[[2.5, 2.6],[2.7,2.8]]],[[[2.1, 2.2],[2.3,2.4]],[[2.5, 2.6],[2.7,2.8]]]])
b = a.view(a.shape[0] * a.shape[1], a.shape[2], a.shape[3], a.shape[4])
c = [a1, a2]
d = torch.stack(c, dim=0)
e = d.view(d.shape[0] * d.shape[1], d.shape[2], d.shape[3], d.shape[4])
print(b)
print(d)