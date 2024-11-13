import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from Model.utils import Transform
from Model.PatchCraft.filters import apply_filters
from Model.PatchCraft.patch_generator import smash_reconstruct

Image.MAX_IMAGE_PIXELS = None


class ImgDataset(Dataset):
    def __init__(self, img_label, train=True):
        """
        :param img: 图像路径
        :param label: 0为真实图像，1为AI生成图像
        """
        self.data = img_label
        self.size = len(img_label)
        self.transform = Transform(train=train)

    def __getitem__(self, idx):
        labels = [0, 0]
        labels[self.data[idx][1]] = 1
        img = Image.open(self.data[idx][0])
        img, img_tensor = self.transform(img)
        rt, pt = smash_reconstruct(img)
        rt, pt = apply_filters(rt), apply_filters(pt)
        return img_tensor, labels, rt, pt

    def __len__(self):
        return self.size


def collate_fn(batch):
    imgs, labels, rts, pts = [], [], [], []
    for img, label, rt, pt in batch:
        imgs.append(img)
        labels.append(label)
        rts.append(rt)
        pts.append(pt)

    labels = np.array(labels, dtype='float32')  # (batch_size, 2)
    rts = np.array(rts, dtype='float32')[:, None]  # (batch_size, 1, 512, 512)
    pts = np.array(pts, dtype='float32')[:, None]  # (batch_size, 1, 512, 512)
    return torch.stack(imgs), torch.tensor(labels), torch.tensor(rts), torch.tensor(pts)
