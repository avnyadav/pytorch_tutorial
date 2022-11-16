from logging import RootLogger
from torchvision import datasets
from torchvision.transforms import ToTensor
from typing import List
from torch import nn
import torch
from typing import Callable

torch.from_numpy()

def dataset() -> List[ToTensor]:
    dataset = []
    training_dataset = datasets.FashionMNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )
    testing_dataset = datasets.FashionMNIST(
        root="data", test=True, download=True, transform=ToTensor()
    )

    return dataset[training_dataset, testing_dataset]


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)



if __name__ == '__main__':
    train, test = dataset()
