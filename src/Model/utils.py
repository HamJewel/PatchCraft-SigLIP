import torch
import torch.nn.functional as F
import torchvision.transforms as tf


def get_loss(logits, labels, reduction='mean'):
    log_p = F.logsigmoid(logits)  # (N, N)
    log_not_p = F.logsigmoid(-logits)  # (N, N)
    nll = -torch.sum(labels * log_p + (1. - labels) * log_not_p, dim=-1)  # (N, )
    return nll.mean() if reduction == 'mean' else nll


def predict(logits, raw=False):
    probs = F.sigmoid(logits)
    return probs if raw else probs.argmax(-1)


def _convert_to_rgb(image):
    return image.convert('RGB')


class Transform:
    def __init__(self, size=(512, 512), train=True):
        if train:
            self.tf1 = tf.Compose([
                tf.RandomResizedCrop(size=size, scale=(0.9, 1.0), ratio=(3 / 4, 4 / 3),
                                     interpolation=tf.InterpolationMode.BICUBIC),
                tf.RandomHorizontalFlip(p=0.5)
            ])
        else:
            self.tf1 = tf.Resize(size=size, interpolation=tf.InterpolationMode.BICUBIC)

        self.tf2 = tf.Compose([
            _convert_to_rgb,
            tf.ToTensor(),
            tf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, img):
        img = self.tf1(img)
        return img, self.tf2(img)
