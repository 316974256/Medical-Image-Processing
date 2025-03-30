import torch
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score, precision_score

torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4486, 0.3167, 0.2259), (0.2229, 0.1545, 0.1068))
])

train_dataset = datasets.ImageFolder(root="data/augmented_resized_V2/train",transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder(root="data/augmented_resized_V2/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(20, 30, 5)
        self.conv4 = torch.nn.Conv2d(30, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(32 * 34 * 34, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 64)
        self.fc5 = torch.nn.Linear(64, 5)

    def forward(self, x, extract_features=False):
        batch_size = x.size(0)
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        feature_map = x
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        if extract_features:
            return feature_map
        return x

def train(epoch):
    running_loss = 0.0
    for batch_idx, (inputs, target) in enumerate(train_loader,0):
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' %(epoch+1, batch_idx+1, running_loss/300))
            running_loss = 0.0

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        scheduler.step()
    torch.save(model.state_dict(), "model.pth")  # 模型默认保存到 CPU

