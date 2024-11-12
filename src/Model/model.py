import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.layers import ConvBlock, MLP
from open_clip import get_tokenizer, create_model_and_transforms


class TextureEncoder(nn.Module):
    def __init__(self, dim):
        super(TextureEncoder, self).__init__()
        block_nums = [6, 6, 4, 4]
        self.rich_texture = ConvBlock(1, dim, activation=nn.Hardtanh())
        self.poor_texture = ConvBlock(1, dim, activation=nn.Hardtanh())
        self.classifier = nn.Sequential(
            ConvBlock(dim, n_blocks=block_nums[0]),
            nn.AvgPool2d(kernel_size=2),
            ConvBlock(dim, n_blocks=block_nums[1]),
            nn.AvgPool2d(kernel_size=2),
            ConvBlock(dim, n_blocks=block_nums[2]),
            nn.AvgPool2d(kernel_size=2),
            ConvBlock(dim, n_blocks=block_nums[3]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1)
        )

    def forward(self, rt, pt):
        x = self.rich_texture(rt) - self.poor_texture(pt)
        y = self.classifier(x)
        return y  # (B, D)


class Model(nn.Module):
    def __init__(self, model_name='ViT-B-16-SigLIP-512', tokenizer_name=None, pretrained=None):
        super(Model, self).__init__()
        self.texture_dim = 32
        self.clip_dim = 768
        self.texture = TextureEncoder(self.texture_dim)
        self.tokenizer = get_tokenizer(tokenizer_name or model_name)
        self.clip, _, _ = create_model_and_transforms(model_name=model_name, pretrained=pretrained)
        self.mlp = MLP(dims=[self.clip_dim + self.texture_dim, self.clip_dim])

    def forward(self, imgs, tokens, rt, pt):
        text_feats = self.clip.encode_text(tokens)
        image_feats = self.clip.encode_image(imgs)
        texture_feats = self.texture(rt, pt)
        image_feats = self.mlp(torch.cat([image_feats, texture_feats], dim=-1))
        image_feats = F.normalize(image_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        logits = image_feats @ text_feats.T * self.clip.logit_scale.exp() + self.clip.logit_bias
        return logits
