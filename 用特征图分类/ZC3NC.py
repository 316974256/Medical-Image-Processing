import torch
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy
from sklearn.metrics import balanced_accuracy_score, precision_score

torch.cuda.empty_cache()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4486, 0.3167, 0.2259), (0.2229, 0.1545, 0.1068))
])

train_dataset = datasets.ImageFolder(root="data/augmented_resized_V2/train",transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

test_dataset = datasets.ImageFolder(root="data/augmented_resized_V2/test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

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

def compute_mean_std(train_dataset, model, device,target_class,test_loader):
    all_features = []
    model.eval()
    class_indices = [i for i, (_, label) in enumerate(train_dataset.samples)
                     if label == target_class]
    class_subset = torch.utils.data.Subset(train_dataset, class_indices)
    class_loader = DataLoader(
        class_subset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(tqdm(class_loader)):
            inputs = inputs.to(device)
            features = model(inputs, extract_features=True)
            features = features.view(features.size(0), -1)
            all_features.append(features.cpu())
            if (batch_idx + 1) % 500 == 0:
                print(f"已处理 {batch_idx+1} 个批次的图片")
    all_features = torch.cat(all_features, dim=0)
    distance_matrix = torch.eye(all_features.size(0))
    total_blocks = (all_features.size(0) // 100 + (1 if all_features.size(0) % 100 != 0 else 0))
    total_iterations = total_blocks ** 2
    with tqdm(total=total_iterations, desc="计算距离矩阵") as pbar:
        for i in range(0, all_features.size(0), 100):
            for j in range(0, all_features.size(0), 100):
                if i > j:
                    pbar.update(1)
                    continue
                chunk_i = all_features[i:i + 100]  # 形状: (chunk_size, feature_dim)
                chunk_j = all_features[j:j + 100]
                diff = chunk_i.unsqueeze(1) - chunk_j.unsqueeze(0)  # 广播减法
                chunk_dist = torch.sqrt((diff ** 2).sum(dim=2))
                distance_matrix[i:i + 100, j:j + 100] = chunk_dist
                if i != j:
                    distance_matrix[j:j + 100, i:i + 100] = chunk_dist.t()
                pbar.update(1)
                pbar.set_postfix({
                    '当前分块': f"{i}-{min(i + 100, all_features.size(0))}×{j}-{min(j + 100, all_features.size(0))}",
                    '矩阵填充进度': f"{distance_matrix.nonzero().size(0) / distance_matrix.numel():.1%}"
                })

    mean_distance = distance_matrix.mean()
    std_distance = torch.sqrt(((distance_matrix - mean_distance) ** 2).mean())
    return mean_distance, std_distance,all_features

def compute_z_score(mean_distance,std_distance,mean_distance_1):
    z_score = abs((mean_distance-mean_distance_1)/std_distance)
    return z_score

def compute_mean(all_features,test_feat):
    all_features = all_features.to(device)
    test_feat = test_feat.to(device)
    diff = all_features - test_feat
    squared_diff = diff ** 2
    distances = torch.sqrt(squared_diff.sum(dim=1))
    mean = distances.mean()
    return mean

def test():
    z_score_0 = compute_z_score(mean_distance_0, std_distance_0, mean_0)
    z_score_1 = compute_z_score(mean_distance_1, std_distance_1, mean_1)
    z_score_2 = compute_z_score(mean_distance_2, std_distance_2, mean_2)
    z_score_3 = compute_z_score(mean_distance_3, std_distance_3, mean_3)
    z_score_4 = compute_z_score(mean_distance_4, std_distance_4, mean_4)
    z_scores = {
        "0": z_score_0,
        "1": z_score_1,
        "2": z_score_2,
        "3": z_score_3,
        "4": z_score_4
    }
    min_class = min(z_scores, key=lambda k: z_scores[k])
    return min_class

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

if __name__ == '__main__':
    model.load_state_dict(torch.load('model.pth',weights_only=True))
    mean_distance_0, std_distance_0,all_features_0 = compute_mean_std(train_dataset, model, device,0,test_loader)
    mean_distance_1, std_distance_1,all_features_1 = compute_mean_std(train_dataset, model, device,1,test_loader)
    mean_distance_2, std_distance_2,all_features_2 = compute_mean_std(train_dataset, model, device,2,test_loader)
    mean_distance_3, std_distance_3,all_features_3 = compute_mean_std(train_dataset, model, device,3,test_loader)
    mean_distance_4, std_distance_4,all_features_4 = compute_mean_std(train_dataset, model, device,4,test_loader)
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            all_labels.extend(labels.cpu().numpy())
            features_1 = model(inputs, extract_features=True)
            features_1 = features_1.view(features_1.size(0), -1)
            for test_feat in features_1:
                mean_0 = compute_mean(all_features_0, test_feat)
                mean_1 = compute_mean(all_features_1, test_feat)
                mean_2 = compute_mean(all_features_2, test_feat)
                mean_3 = compute_mean(all_features_3, test_feat)
                mean_4 = compute_mean(all_features_4, test_feat)
                predicted = test()
                all_predicted.append(int(predicted))
            if (batch_idx + 1) % 500 == 0:
                print(f"已处理 {batch_idx+1} 个批次的图片")

    balanced_acc = balanced_accuracy_score(all_labels, all_predicted)
    weighted_p = precision_score(all_labels, all_predicted, average="weighted")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Weighted Precision: {weighted_p:.4f}")
