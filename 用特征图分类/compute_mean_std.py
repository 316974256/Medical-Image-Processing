import torch
from torchvision import datasets, transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0,0,0), (1,1,1,))
])
train_dataset = datasets.ImageFolder(root="data/augmented_resized_V2/train",transform=transform)
def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    mean = 0.0
    std = 0.0
    total_samples = 0

    for batch_idx, (images, _) in enumerate(tqdm(loader)):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)  # 展平

        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_samples += batch_samples

        if (batch_idx + 1) % 500 == 0:
            print(f"已处理 {total_samples} 张图片")

    mean /= total_samples
    std /= total_samples

    return mean, std

if __name__ == '__main__':
    mean, std = compute_mean_std(train_dataset)
    print(mean, std)

'''
tensor([0.4486, 0.3167, 0.2259]) tensor([0.2229, 0.1545, 0.1068])
'''