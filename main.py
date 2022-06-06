import torch
import torchvision
import torchvision.transforms as trfs
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image

BATCH_SIZE = 64

def getMNISTDataLoader():
    trainset = torchvision.datasets.MNIST(
        root="./MNIST",
        train=True,
        transform=trfs.ToTensor(),
        target_transform=trfs.Lambda(lambda x : 
            torch.zeros(10).scatter(0, torch.tensor(x), value=1)
        )
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    testset = torchvision.datasets.MNIST(
        root="./MNIST",
        train=False,
        transform=trfs.ToTensor(),
        target_transform=trfs.Lambda(lambda x : 
            torch.zeros(10).scatter(0, torch.tensor(x), value=1)
        )
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=8
    )
    return trainloader, testloader

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512), #input layer
            torch.nn.ReLU(), #activation
            torch.nn.Linear(512, 512), # hidden layer
            torch.nn.ReLU(), #activation
            torch.nn.Linear(512, 512), # hidden layer
            torch.nn.ReLU(), #activation
            torch.nn.Linear(512, 10) #output layer
        )

    def forward(self, inp):
        inp = self.flatten(inp)
        out = self.linear_relu_stack(inp)
        return out

    def train(self, dataloader, epochs, learning_rate, device="cuda"):
        loss_f = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        nBatches = int(len(dataloader.dataset)/BATCH_SIZE)
        for epoch in range(epochs):
            for currBatch, (data, labels) in enumerate(dataloader):
                data = data
                # Feed forward
                output = self(data.cuda())
                loss = loss_f(output, labels.cuda())

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if currBatch % 100 == 0:
                    print(f"[{epoch}/{epochs}][{currBatch}/{nBatches}] loss : {loss.item()}")
            pass

    def test(self, testloader):
        correctguesses = [0]*10
        total = [0]*10
        totalinpts = len(testloader.dataset)
        for nTest, (data, label) in enumerate(testloader):
            expected = label.argmax(1).item()
            total[expected] += 1
            guess = self.evaluate(data.cuda())
            if guess == expected:
                correctguesses[expected] += 1
            if nTest % 500 == 0:
                print(f"[Test {nTest}/{totalinpts}] Current accuracy : {(sum(correctguesses)/(nTest+1))*100}%")
        print(f"Final accuracy : {(sum(correctguesses)/(totalinpts))*100}%")
        print([(b / m)*100 for b,m in zip(correctguesses, total)])

    def evaluate(self, data):
        out = self(data)
        out = torch.nn.Softmax(dim=1)(out)
        pick = out.argmax(1)
        return pick.item()

trainloader, testloader = getMNISTDataLoader()

if sys.argv[1] == "train":
    model = NeuralNetwork().cuda()
    try:
        model.train(trainloader, 150, 1e-4)
    except KeyboardInterrupt:
        pass
    torch.save(model.state_dict(), "model.pth")
elif sys.argv[1] == "test":
    model = NeuralNetwork().to("cuda")
    model.load_state_dict(torch.load("model.pth"))
    model.test(testloader)
elif sys.argv[1] == "resume":
    model = NeuralNetwork().to("cuda")
    model.load_state_dict(torch.load("model.pth"))
    try:
        model.train(trainloader, 100, 1e-5)
    except KeyboardInterrupt:
        pass
    torch.save(model.state_dict(), "model.pth")
elif sys.argv[1] == "use":
    model = NeuralNetwork().to("cuda")
    model.load_state_dict(torch.load("model.pth"))
    # model.eval()
    while True:
        input()
        with Image.open(sys.argv[2]) as img:
            img = trfs.Grayscale()(img)
            inputtensor = trfs.ToTensor()(img).unsqueeze(0).to("cuda")
            # print(inputtensor.shape)
            res = model.evaluate(inputtensor)
            print(res, end="")



# #random 28*28 tensor for input
# randomImage = torch.rand(1, 28, 28, device="cuda")
# model = NeuralNetwork().to("cuda")

# #feed random input to model and get output
# out = model(randomImage)
# print(out)

# #softmax maps outputs from 0 to 1 so that their sums is equal
# #to 1 (good for probabilities)

# predicted = out.argmax(1)
# print("Predicted :", predicted.item())