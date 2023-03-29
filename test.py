import torch
# caole

a = 0.0
if 0.0 == False:
    print("yiyang loves zhangyitian") 
# Example 1
t1 = torch.randn(3,1,4)
t2 = torch.randn(3,1,4)

stacked1 = torch.stack([t1, t2], dim=0).squeeze()
print(stacked1.shape)

stacked2 = torch.stack([t1, t2], dim=1)
print(stacked2.shape)
stacked2 = stacked2.squeeze()
print(stacked2.shape)
# Output:
# torch.Size([2, 3, 1, 4])
# torch.Size([3, 2, 1, 4])
# torch.Size([3, 2, 4])

print("shuuuuuuuuuuuuuuuuuuuu")
t3 = torch.randn(3,3,4)
print(t3[:,1,:].shape)
st = torch.stack([t3[:,1,:], t3[:,2,:]], dim=1)
print(st.shape)
print(torch.__version__)


print("shuuuuuuuuuuuuuuuuuuuu")
zyt = torch.nn.Linear(4,5)
d3 = torch.randn(3,4,4)

print(zyt(d3).shape)
