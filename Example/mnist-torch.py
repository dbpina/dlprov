import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import EMNIST
import os
from datetime import datetime  
import time

from dfa_lib_python.dataflow import Dataflow
from dfa_lib_python.transformation import Transformation
from dfa_lib_python.attribute import Attribute
from dfa_lib_python.attribute_type import AttributeType
from dfa_lib_python.set import Set
from dfa_lib_python.set_type import SetType
from dfa_lib_python.task import Task
from dfa_lib_python.dataset import DataSet
from dfa_lib_python.element import Element
from dfa_lib_python.task_status import TaskStatus   


dataflow_tag = "example"    
exec_tag = dataflow_tag + "-" + str(datetime.now())

df = Dataflow(dataflow_tag, predefined=True)
df.save()

t1 = Task(1, dataflow_tag, exec_tag, "LoadData")
t1_1 = Task(2, dataflow_tag, exec_tag, "RandomHorizontal", dependency = t1)
t1_2 = Task(3, dataflow_tag, exec_tag, "Normalize", dependency = t1_1)
t2 = Task(4, dataflow_tag, exec_tag, "SplitData", dependency = t1_2)
t3 = Task(5, dataflow_tag, exec_tag, "Train", dependency = t2)
t4 = Task(6, dataflow_tag, exec_tag, "Test", dependency = [t2,t3])  

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

url_dataset = EMNIST.url

tf1_input = DataSet("iInputDataset", [Element(["EMNIST", url_dataset])])
t1.add_dataset(tf1_input)
t1.begin()  

# Load and transform MNIST dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),  # Flip the image horizontally
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = EMNIST(root='./data', split='mnist', train=True, download=True, transform=transform)
testset = EMNIST(root='./data', split='mnist', train=False, download=True, transform=transform)   


tf1_output = DataSet("oLoadData", [Element([os.path.join(os.getcwd()) + "./data/EMNIST/raw"])])
t1.add_dataset(tf1_output)
t1.end()  

DATASET_DIR = "./data/EMNIST/raw"

random_output = DataSet("oRandomHorizontal", [Element([DATASET_DIR])])
t1_1.add_dataset(random_output) 
t1_1.end()  

normalization_output = DataSet("oNormalize", [Element([DATASET_DIR])])
t1_2.add_dataset(normalization_output) 
t1_2.end()    

# dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split dataset into train (80%) and validation (20%)
train_size = int(0.8 * len(trainset))
val_size = len(trainset) - train_size

tf2_input = DataSet("iSplitConfig", [Element([f"{train_size}", f"{val_size}", f"{len(testset)}"])])
t2.add_dataset(tf2_input)
t2.begin() 

trainset, valset = random_split(trainset, [train_size, val_size])    

# Create data loaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
valloader = DataLoader(valset, batch_size=64, shuffle=False)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

tf2_train_output = DataSet("oTrainSet", [Element([trainloader.batch_size])])
t2.add_dataset(tf2_train_output)
t2.save()

tf2_train_output = DataSet("oValSet", [Element([valloader.batch_size])])
t2.add_dataset(tf2_train_output)
t2.save()

tf2_test_output = DataSet("oTestSet", [Element([testloader.batch_size])])
t2.add_dataset(tf2_test_output)
t2.end()      

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


optimizer_name = optimizer.__class__.__name__
learning_rate = optimizer.param_groups[0]['lr']  # Get learning rate from optimizer
num_layers = len(list(model.children()))  # Count the number of layers
num_epochs = 2
batch_size = 64

tf3_input = DataSet("iTrain", [Element([optimizer_name, learning_rate, num_epochs, batch_size, num_layers])])
t3.add_dataset(tf3_input)
t3.begin() 

# Function to calculate accuracy and loss
def evaluate(loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return total_loss / len(loader), 100 * correct / total

# Training loop
for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(trainloader)
    train_acc = 100 * correct / total
    val_loss, val_acc = evaluate(valloader)
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')                 

    elapsed_time = (time.time() - start_time)  

    tf3_output = DataSet("oTrain", [
        Element([
                timestamp,
                elapsed_time,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                epoch + 1
            ])
        ])

    t3.add_dataset(tf3_output)
    t3.save()     

trained_path = os.path.join(os.getcwd(), "modelv1.pth")
torch.save(model.state_dict(), trained_path)

tf3_output_model = DataSet("oTrainedModel", [Element(["modelv1.pth", str(trained_path)])])
t3.add_dataset(tf3_output_model)
t3.end()   

t4.begin()       

# Final evaluation on test set
test_loss, test_acc = evaluate(testloader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')

testing_output = DataSet("oTest", [Element([test_loss, test_acc])])

t4.add_dataset(testing_output) 
t4.end()   