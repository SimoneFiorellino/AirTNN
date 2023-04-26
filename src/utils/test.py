import torch

batch_size = 64
num_nodes = 100
num_classes = 10

out = torch.randn(batch_size, num_nodes, num_classes)
target = torch.randint(0, num_classes, (batch_size,))
# for each batch each node, we have a probability distribution over the classes
# apply a softmax to get a probability distribution
x = torch.softmax(out, dim=-1)
# apply the argmax to get the most probable class
x = torch.argmax(x, dim=-1)
print(x.shape)
print(x)
# for each batch, given the decision of each node, we take the most frequent decision
pred = torch.mode(x, dim=1).values
print(pred.shape)
print(pred)
# now we want to compute the accuracy between pred and target
acc = torch.sum(pred == target).float() / batch_size
print("acc", acc)

# now we want to compute the loss between pred and target
logits = torch.mean(out, dim=1)
loss = torch.nn.functional.cross_entropy(logits, target)
print("loss", loss)
logits = torch.log_softmax(logits, dim=-1)
print(logits.shape)
loss = torch.nn.functional.cross_entropy(logits, target)
print("loss", loss)

# now we want to compute the accuracy between pred and target
acc = torch.sum(pred == target).float() / batch_size
print("acc", acc)
