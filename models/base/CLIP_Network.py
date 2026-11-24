# Reconstructed from the authors' compiled file models/base/__pycache__/CLIP_Network.cpython-38.pyc (the .py was not
# released). Zero-shot CLIP baseline of the paper's Figure 3 (open_clip ViT-B/16, LAION-2B). Checked against the bytecode.
import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar as resnet18_cifar
from utils import identify_importance
import numpy as np, copy, open_clip as clip
from open_clip import tokenizer

class CLIP(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        self.encoder, _, self.preprocess = clip.create_model_and_transforms("ViT-B/16", pretrained="laion2b_s34b_b88k")
        self.seen_classes = args.base_class

    def forward_metric(self, x):
        x = self.encode(x)
        x = self.get_logits(x)
        return x

    def encode(self, x):
        (image_input, text_inputs) = x
        image_features = self.encoder.encode_image(image_input)
        text_features = self.encoder.encode_text(text_inputs)
        return (
         image_features, text_features)

    def forward(self, input):
        input = self.forward_metric(input)
        return input

    def get_logits(self, x):
        (image_features, text_features) = x
        image_features /= image_features.norm(dim=(-1), keepdim=True)
        text_features /= text_features.norm(dim=(-1), keepdim=True)
        similarity = 100.0 * image_features @ text_features.T
        return similarity

    def train_inc(self):
        print("[Train Incrermental Session]: TBA..")

