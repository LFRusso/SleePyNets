import torch
import torch.optim as optim
import torch.nn.functional as F

def get_activations(model, loader,device_name="cpu"):
    for key in model.max_activations:
        model.max_activations[key] = 0
    device = torch.device(device_name)
    model.eval()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        model(data)

def train(model, train_loader, epoch, device_name="cpu"):
    device = torch.device(device_name)
    optimizer=optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
    
    model.train()
    L = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        L.append(loss)
    return L

def test(model, test_loader, device_name="cpu"):
    device = torch.device(device_name)
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def mean_activity(loader):
    mean = torch.zeros([1,28,28]) # Same dimensions as the inputs
    for images, _ in loader:
        mean += torch.sum(images, axis=0)
    mean /= len(loader.dataset)
    return mean
