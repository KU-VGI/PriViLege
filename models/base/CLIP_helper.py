# Reconstructed from the authors' compiled file models/base/__pycache__/CLIP_helper.cpython-38.pyc (the .py was not
# released). Zero-shot CLIP baseline of the paper's Figure 3 (open_clip ViT-B/16, LAION-2B). Checked against the bytecode.
import time
from .CLIP_Network import CLIP
from utils import *
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel
import numpy as np, open_clip as clip
from open_clip import tokenizer

def validate_path(path):
    if os.path.exists(path):
        pass
    else:
        print("create folder:", path)
        os.makedirs(path)


def base_train(model, trainloader, text_labels, optimizer, scheduler, epoch, args):
    print("[Base Train]")
    tl = Averager_Loss()
    ta = Averager()
    model = model.train()
    tqdm_gen = tqdm(trainloader)
    for i, batch in enumerate(tqdm_gen, 1):
        data, train_label = [_.cuda() for _ in batch]
        train_text = text_labels
        train_text = torch.cat([tokenizer.tokenize(f"a photo of a {text_labels[lab_idx]}") for lab_idx in train_label]).cuda()
        logits = model((data, train_text))
        loss_r = F.cross_entropy(logits, train_label)
        acc = count_acc(logits, train_label)
        total_loss = loss_r
        lrc = scheduler.get_last_lr()[0]
        tl.add(total_loss.item(), len(train_label))
        ta.add(acc, len(train_label))
        tqdm_gen.set_description("Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}".format(epoch, lrc, total_loss.item(), ta.item()))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    tl = tl.item()
    ta = ta.item()
    return (
     tl, ta)


def test(model, testloader, text_labels, epoch, args, session):
    test_class = args.base_class + session * args.way
    model = model.eval()
    vl = Averager_Loss()
    va = Averager()
    va_base = Averager()
    va_new = Averager()
    va_base_given_new = Averager()
    va_new_given_base = Averager()
    print("\t\t\t[Test Phase] Session: {}".format(session))
    test_text = torch.cat([tokenizer.tokenize(f"a photo of a {t_lab}") for t_lab in text_labels]).cuda()
    with torch.no_grad():
        tqdm_gen = tqdm(testloader)
        for (i, batch) in enumerate(tqdm_gen, 1):
            (data, test_label) = [_.cuda() for _ in batch]
            if session == 0:
                logits = model((data, test_text))
            else:
                logits = model((data, test_text))
            max_logits = logits.max(dim=1)[0].mean()
            min_logits = logits.min(dim=1)[0].mean()
            loss = F.cross_entropy(logits, test_label)
            acc = count_acc(logits, test_label)
            base_idxs = test_label < args.base_class
            if torch.any(base_idxs):
                acc_base = count_acc(logits[base_idxs, :args.base_class], test_label[base_idxs])
                acc_base_given_new = count_acc(logits[base_idxs, :], test_label[base_idxs])
                va_base.add(acc_base, len(test_label[base_idxs]))
                va_base_given_new.add(acc_base_given_new, len(test_label[base_idxs]))
            new_idxs = test_label >= args.base_class
            if torch.any(new_idxs):
                acc_new = count_acc(logits[new_idxs, args.base_class:], test_label[new_idxs] - args.base_class)
                acc_new_given_base = count_acc(logits[new_idxs, :], test_label[new_idxs])
                va_new.add(acc_new, len(test_label[new_idxs]))
                va_new_given_base.add(acc_new_given_base, len(test_label[new_idxs]))
            vl.add(loss.item(), len(test_label))
            va.add(acc, len(test_label))

        vl = vl.item()
        va = va.item()
        va_base = va_base.item()
        va_new = va_new.item()
        va_base_given_new = va_base_given_new.item()
        va_new_given_base = va_new_given_base.item()
    print("epo {}, test, loss={:.4f} acc={:.4f}".format(epoch, vl, va))
    print("base only accuracy: {:.4f}, new only accuracy: {:.4f}".format(va_base, va_new))
    print("base acc given new : {:.4f}".format(va_base_given_new))
    print("new acc given base : {:.4f}".format(va_new_given_base))
    logs = dict(num_session=(session + 1), acc=va, base_acc=va_base, new_acc=va_new, base_acc_given_new=va_base_given_new, new_acc_given_base=va_new_given_base)
    return (
     vl, va, logs)


def build_label_embedding(train_set, session, text_labels):
    new_classes = np.unique(train_set.targets)
    classes = np.unique(train_set.classes)
    print("Number of classes:", len(classes))
    classes_int = np.unique(train_set.targets)
    print("classes_int:", classes_int)
    print("new classes for session {} : {} \n".format(session, classes[classes_int]))
    words_embed = []
    with torch.no_grad():
        for cls in classes[classes_int]:
            if cls not in text_labels:
                text_labels.append(cls)

    return text_labels

