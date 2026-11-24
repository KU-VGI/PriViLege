# Reconstructed from the authors' compiled file models/base/__pycache__/ViT_Network_CLIP.cpython-39.pyc
# (the .py was not released). PriViLege on a pre-trained CLIP ViT-B/16 (LAION-2B), as in Figure 3
# of the paper: the vision encoder (official open_clip + prompts, models/CLIP_prompt.py) carries the
# B-Prompt and VL-Prompt; the CLIP text encoder stays frozen and gives the class-name embeddings.
# Apart from __init__ (official encoder) and the marked FIX, every function matches the bytecode.
import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance
import numpy as np
import copy
import timm
from models.CLIP_prompt import PromptedVisionTransformer
import open_clip as clip

# The authors' file switched on torch.autograd.set_detect_anomaly(True) here. It is a debugging aid that
# does not change results but slows every training step, so it is off for the reproduction runs.


class ViT_MYNET_CLIP(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100']:
            self.num_features = 768
        if self.args.dataset in ['mini_imagenet']:
            self.num_features = 768
        if self.args.dataset == 'cub200':
            self.num_features = 768

        # Official open_clip CLIP ViT-B/16 (LAION-2B). The text encoder stays frozen; a copy of its vision
        # encoder gets the PriViLege prompts. (The authors instead loaded these weights into their own copy
        # of OpenAI's VisionTransformer with strict=False; see models/CLIP_prompt.py.)
        self.text_encoder, _, _ = clip.create_model_and_transforms('ViT-B/16', pretrained='laion2b_s34b_b88k')
        self.encoder = PromptedVisionTransformer(copy.deepcopy(self.text_encoder.visual))
        self.tokenizer = clip.get_tokenizer('ViT-B-16')

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.seen_classes = args.base_class

    def update_seen_classes(self, new_classes):
        print('new classes for this session:\n', new_classes)
        self.mask = torch.zeros(self.args.num_classes, device='cuda')
        self.mask[:self.seen_classes] = -torch.inf
        self.seen_classes += len(new_classes)

    def forward_fc(self, image_features, label, word_text):
        out = {}
        with torch.no_grad():
            logit_head = torch.cat([self.tokenizer(f'a photo of a {c}') for c in word_text]).cuda()
            text_heads = self.text_encoder.encode_text(logit_head)
        text = torch.cat([self.tokenizer(f'a photo of a {c}') for c in word_text[label.cpu()]]).cuda()
        text_features = self.text_encoder.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        out['image_sim'] = image_features @ image_features.T
        out['text_sim'] = text_features @ text_features.T
        out['logits'] = image_features @ text_features.T
        out['logit_pred'] = image_features @ text_heads.T
        return out

    def encode(self, x):
        x = self.encoder(x, VL_tuning=False, B_tuning=False)
        return x[:, 0]

    def prompt_encode(self, input):
        prompt_embed = {}
        x = self.encoder(input, VL_tuning=True, B_tuning=True)
        cls_embed = x[:, 0]
        prompt_embed['Vision'] = x[:, 1]
        prompt_embed['Language'] = x[:, 2]
        return cls_embed, prompt_embed

    def forward(self, input_data, word_info=None, prompt_feat=False, base=False, query=False):
        input, label = input_data
        if base:
            embedding = self.prompt_encode(input)
            cls_embed, prompt_embed = embedding
            logit = self.forward_fc(0.5 * (prompt_embed['Vision'] + cls_embed), label, word_info['label_text'])
            return logit, cls_embed, prompt_embed
        if query:
            q_feat = self.encode(input)
            return q_feat
        if self.mode == 'encoder':
            embedding = self.prompt_encode(input)
            if prompt_feat:
                cls_embed, prompt_embed = embedding
                return cls_embed, prompt_embed
            return embedding
        elif self.mode != 'encoder':
            x = self.prompt_encode(input)
            cls_embed, prompt_embed = x
            logit = self.forward_fc(0.5 * (prompt_embed['Vision'] + cls_embed), label, word_info['label_text'])
            return logit
        else:
            raise ValueError('Unknown mode')

    def cross_entropy(self, preds, targets, reduction='none'):
        loss = F.cross_entropy(preds, targets, reduction='none')
        if reduction == 'none':
            return loss
        elif reduction == 'mean':
            return loss.mean()

    def clip_loss(self, out, targets):
        logits = out['logits']
        images_similarity = out['image_sim']
        texts_similarity = out['text_sim']
        label = torch.arange(targets.shape[0]).cuda()
        texts_loss = self.cross_entropy(logits, label, reduction='none')
        images_loss = self.cross_entropy(logits.T, label, reduction='none')
        loss = (images_loss + texts_loss) / 2.0
        return loss.mean()

    def train_inc(self, dataloader, epochs, session, class_list, word_info, query_info):
        print('[Session: {}]'.format(session))
        # FIX (not in the authors' final compiled version, where update_fc_avg is empty and this call is
        # missing): without the new classes' prototypes, knowledge_boosting (-SKD) fails in session 1 with
        # "Target 60 is out of bounds". Restored from the authors' earlier version of this class.
        self.update_fc_avg(dataloader, class_list, query_info)

        for idx, batch in enumerate(dataloader):
            data_imgs, data_label = [_.cuda() for _ in batch]
            optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.args.lr_new)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

            word_cur_embed = word_info['cur_embed'].cuda()
            word_embed = word_info['embed'].cuda()
            for epoch in range(epochs):
                self.train()
                cls_feat, prompt_feat = self.prompt_encode(data_imgs)
                outs = self.forward_fc(0.5 * (prompt_feat['Vision'] + cls_feat), data_label, word_info['label_text'])
                loss_ce = self.clip_loss(outs, data_label)
                logits = outs['logit_pred']
                if self.args.SKD:
                    loss_kb = self.knowledge_boosting(prompt_feat['Language'], word_embed, query_info, data_label)
                    loss = loss_ce + loss_kb
                else:
                    loss = loss_ce

                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()
                pred = torch.argmax(logits, dim=1)
                acc = (pred == data_label).sum().item() / data_label.shape[0] * 100.
                if self.args.SKD:
                    print(f'[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} loss_kb:{loss_kb.item():.4f} ACC: {acc}')
                else:
                    print(f'[{epoch}/{epochs}] Loss_CE:{loss_ce.item():.4f} ACC: {acc}')

    def triplet(self, cls_embed, vision_embed, query_info, train_label):
        P_head = query_info['proto'].clone().cuda()

        cls_logit = F.linear(cls_embed, P_head)
        cls_gt = F.cross_entropy(cls_logit, train_label, reduction='none')
        vis_logit = F.linear(vision_embed, P_head)
        vis_gt = F.cross_entropy(vis_logit, train_label, reduction='none')
        cls_vis = F.cross_entropy(cls_logit, torch.softmax(vis_logit, dim=1), reduction='none')

        loss_tri = -1 * (cls_vis.mean() / (cls_vis.mean() + cls_gt.mean() + vis_gt.mean()) + 1e-6).log()
        return loss_tri

    def knowledge_boosting(self, lang_embed, word_embed, query_info, label):
        P_head = query_info['proto'].clone().cuda()
        T = 2.
        lang_logit = F.linear(lang_embed, P_head)
        loss_seman = F.cross_entropy(lang_logit, label)
        loss_kd = F.kl_div(F.log_softmax(lang_embed / T, dim=1), F.softmax(word_embed[label] / T, dim=1), reduction='batchmean')
        loss = loss_kd + 0.2 * loss_seman
        return 0.5 * loss

    def update_fc_avg(self, dataloader, class_list, query_info):
        # FIX: body restored from the authors' earlier version (ViT_Network_CLIP.cpython-38.pyc); it adds the new
        # classes' prototypes (mean CLS feature) to query_info for the SKD loss. That version also copied them into
        # self.fc, which is left out: fc is 768-wide while these CLIP features are 512, and CLIP classifies
        # with the text heads, not fc.
        self.eval()
        query_p = []
        with torch.no_grad():
            for batch in dataloader:
                data_imgs, label = [_.cuda() for _ in batch]
                cls_embed = self.encode(data_imgs).detach()

            for class_index in class_list:
                data_index = (label == class_index).nonzero().squeeze(-1)
                embedding = cls_embed[data_index]
                proto = embedding.mean(0)
                query_p.append(proto)

            query_p = torch.stack(query_p)
        query_info['proto'] = torch.cat([query_info['proto'], query_p.cpu()])
        self.train()

    def get_logits(self, x, fc):
        return fc(x)
