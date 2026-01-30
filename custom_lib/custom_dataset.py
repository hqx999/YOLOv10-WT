import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        """
        Args:
            image_dir (str): 图像文件夹路径
            label_dir (str): 标签文件夹路径（每个标签对应一个文本文件）
            transform: 图像预处理操作（如调整大小、归一化等）
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = []

        # 遍历图像文件，根据标签是否为空进行过滤
        for f in sorted(os.listdir(image_dir)):
            if f.lower().endswith(('.jpg', '.png')):
                # 构造对应标签的路径（假设图片和标签同名，仅后缀不同）
                label_file = os.path.join(label_dir, f.replace('.jpg', '.txt').replace('.png', '.txt'))
                # 如果标签文件存在且内容非空，则加入数据集中
                if os.path.exists(label_file):
                    with open(label_file, 'r') as fp:
                        content = fp.read().strip()
                    if content:  # 非空标签
                        self.image_paths.append(f)
                # 如果标签文件不存在或为空，则跳过该图像

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_name = self.image_paths[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 加载图像
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # 加载标签（格式为：class_id center_x center_y width height，每行一个目标）
        labels = self.load_labels(label_path)

        if self.transform:
            image = self.transform(image)
        
        labels = torch.tensor(labels, dtype=torch.float32)
        return image, labels

    def load_labels(self, label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            if line.strip():
                labels.append(list(map(float, line.strip().split())))
        return labels
