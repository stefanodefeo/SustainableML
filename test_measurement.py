#Modified from "Train CIFAR10 with PyTorch" https://github.com/kuangliu/pytorch-cifar
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from resnet import ResNet18
from measurement import EnergyMeasurementTool, AsynchronousMeasurementTool, LayerwiseMeasurementTool

device = 'cuda' if torch.cuda.is_available() else 'cpu'

num_epochs = 1
batch_size = 128

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False, num_workers=2)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        print('train:',batch_idx+1,'/',len(trainloader),100.*correct/total,end='\r')

    print("")

def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            print('test:',batch_idx+1,'/',len(testloader),100.*correct/total,end='\r')

    print("")
    return 100.*correct/total

model = ResNet18()
criterion = nn.CrossEntropyLoss()
model = model.to(device)
optimizer = optim.Adam(model.parameters(),lr=0.001)

trackerE = EnergyMeasurementTool(out_file='test_energy_measurement_tool.csv',gpu_num=0)
for epoch in range(num_epochs):
    trackerE.start()
    train(epoch)
    test(epoch)
    trackerE.stop()

trackerA = AsynchronousMeasurementTool(out_file='test_asynchronous_measurement_tool.csv',gpu_num=0)
trackerA.start(1)
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)
trackerA.stop()

trackerL = LayerwiseMeasurementTool('test_layerwise_measurement_tool.csv',0)
model = trackerL.start(model)
for epoch in range(num_epochs):
    train(epoch)
trackerL.stop()