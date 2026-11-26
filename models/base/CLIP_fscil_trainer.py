# Reconstructed from the authors' compiled file models/base/__pycache__/CLIP_fscil_trainer.cpython-38.pyc (the .py was not
# released). Zero-shot CLIP baseline of the paper's Figure 3 (open_clip ViT-B/16, LAION-2B). Checked against the bytecode.
from .base import Trainer
import os.path as osp
import torch.nn as nn
from .parallel import DataParallelModel, DataParallelCriterion
import copy
from copy import deepcopy
import pandas as pd
from os.path import exists as is_exists
from .CLIP_helper import *
from utils import *
from dataloader.data_utils import *
from models.switch_module import switch_module as switch_module

class CLIP_FSCILTrainer(Trainer):

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_log_path()
        self.args = set_up_datasets(self.args)
        self.model = CLIP((self.args), mode=(self.args.base_mode))
        self.val_model = CLIP((self.args), mode=(self.args.base_mode))
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()
        del self.val_model   # never used; built only to keep the random-number sequence (see ViT_fscil_trainer)
        self.text_labels = []
        if self.args.model_dir is not None:
            print("Loading init parameters from: %s" % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)["params"]
        else:
            print("random init params")
            if args.start_session > 0:
                print("WARING: Random init weights for new sessions!")
            self.best_model_dict = deepcopy(self.model.state_dict())
        print("##################################################")
        trainable_params = sum((param.numel() for param in self.model.parameters() if param.requires_grad))
        self.init_params = sum((param.numel() for param in self.model.parameters()))
        print("total parameters:", self.init_params)
        print("trainable parameters:", trainable_params)
        print("##################################################")

    def get_optimizer_base(self):
        optimizer = torch.optim.Adam((filter((lambda p: p.requires_grad), self.model.parameters())), lr=(self.args.lr_base))
        if self.args.schedule == "Step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=(self.args.step), gamma=(self.args.gamma))
        elif self.args.schedule == "Milestone":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(self.args.milestones), gamma=(self.args.gamma))
        elif self.args.schedule == "Cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(self.args.epochs_base))
        return (
         optimizer, scheduler)

    def get_dataloader(self, session):
        if session == 0:
            (trainset, trainloader, testloader) = get_base_dataloader((self.args), clip_trsf=None)
        else:
            (trainset, trainloader, testloader) = get_new_dataloader((self.args), session, clip_trsf=None)
        return (trainset, trainloader, testloader)

    def train(self):
        args = self.args
        t_start_time = time.time()
        result_list = [
         args]
        columns = [
         "num_session","acc","base_acc","new_acc","base_acc_given_new","new_acc_given_base"]
        acc_df = pd.DataFrame(columns=columns)
        print("[Start Session: {}] [Sessions: {}]".format(args.start_session, args.sessions))
        for session in range(args.start_session, args.sessions):
            (train_set, trainloader, testloader) = self.get_dataloader(session)
            self.text_labels = build_label_embedding(train_set, session, self.text_labels)
            print("[Session: {}]Current text_labels info: {}".format(session, self.text_labels))
            if session == 0:
                (optimizer, scheduler) = self.get_optimizer_base()
                print("[Base Session Training]")
                print("##################################################")
                trainable_params = sum((param.numel() for param in self.model.parameters() if param.requires_grad))
                print("[Session {}] Trainable parameters: {}".format(session, trainable_params))
                print("##################################################")
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    (tl, ta) = base_train(self.model, trainloader, np.array(self.text_labels), optimizer, scheduler, epoch, args)
                    (tsl, tsa, logs) = test(self.model, testloader, np.array(self.text_labels), epoch, args, session)
                    if tsa * 100 >= self.trlog["max_acc"][session]:
                        self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                        self.trlog["max_acc_epoch"] = epoch
                        save_model_dir = os.path.join(args.save_path, "session" + str(session) + "_max_acc.pth")
                        torch.save(dict(params=(self.model.state_dict())), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print("********A better model is found!!**********")
                        print("Saving model to :%s" % save_model_dir)
                    print("best epoch {}, best test acc={:.3f}".format(self.trlog["max_acc_epoch"], self.trlog["max_acc"][session]))
                    self.trlog["train_loss"].append(tl)
                    self.trlog["train_acc"].append(ta)
                    self.trlog["test_loss"].append(tsl)
                    self.trlog["test_acc"].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append("epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f" % (
                     epoch, lrc, tl, ta, tsl, tsa))
                    print("This epoch takes %d seconds" % (time.time() - start_time), "\nstill need around %.2f mins to finish this session" % ((time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append("Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n".format(session, self.trlog["max_acc_epoch"], self.trlog["max_acc"][session]))
                save_model_dir = os.path.join(args.save_path, "session" + str(session) + "_last_epoch.pth")
                torch.save(dict(params=(self.model.state_dict())), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    best_model_dir = os.path.join(args.save_path, "session" + str(session) + "_max_acc_replace_head.pth")
                    print("Replace the fc with average embedding, and save it to :%s" % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    self.model.module.mode = "avg_cos"
                    (tsl, tsa, logs) = test(self.model, testloader, np.array(self.text_labels), 0, args, session)
                    self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                    print("The new best test acc of base session={:.3f}".format(self.trlog["max_acc"][session]))
            else:
                print("Incremental session: [%d]" % session)
                print("##################################################")
                trainable_params = sum((param.numel() for param in self.model.parameters() if param.requires_grad))
                print("[Session {}] Trainable parameters: {}".format(session, trainable_params))
                print("##################################################")
                self.model.module.mode = self.args.new_mode
                self.model.train()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.eval()
                (tsl, tsa, logs) = test(self.model, testloader, np.array(self.text_labels), 0, args, session)
                acc_df = pd.concat([acc_df, pd.DataFrame([logs])], ignore_index=True)   # pandas 2 removed DataFrame.append
                self.trlog["max_acc"][session] = float("%.3f" % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, "session" + str(session) + "_max_acc.pth")
                self.best_model_dict = deepcopy(self.model.state_dict())
                print("Saving model to :%s" % save_model_dir)
                print("  test acc={:.3f}".format(self.trlog["max_acc"][session]))
                result_list.append("Session {}, test Acc {:.3f}\n".format(session, self.trlog["max_acc"][session]))

        result_list.append("Base Session Best Epoch {}\n".format(self.trlog["max_acc_epoch"]))
        result_list.append(self.trlog["max_acc"])
        print(self.trlog["max_acc"])
        save_list_to_txt(os.path.join(args.save_path, "results.txt"), result_list)
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print("Base Session Best epoch:", self.trlog["max_acc_epoch"])
        print("Total time used %.2f mins" % total_time)
        end_params = sum((param.numel() for param in self.model.module.parameters()))
        print("[Begin] Total parameters: {}".format(self.init_params))
        print("[END] Total parameters: {}".format(end_params))

    def set_save_path(self):
        mode = self.args.base_mode + "-" + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + "-" + "data_init"
        self.args.save_path = "%s/" % self.args.dataset
        if self.args.vit:
            self.args.save_path = self.args.save_path + "%s/" % (self.args.project + "_ViT_Ours")
        else:
            self.args.save_path = self.args.save_path + "%s/" % self.args.project
        self.args.save_path = self.args.save_path + "%s-start_%d/" % (mode, self.args.start_session)
        if self.args.schedule == "Milestone":
            mile_stone = str(self.args.milestones).replace(" ", "").replace(",", "_")[1:-1]
            self.args.save_path = self.args.save_path + "Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d" % (
             self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
             self.args.momentum, self.args.decay, self.args.seed)
        elif self.args.schedule == "Step":
            self.args.save_path = self.args.save_path + "Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d" % (
             self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
             self.args.momentum, self.args.decay, self.args.seed)
        else:
            self.args.save_path = self.args.save_path + "Epo_%d-Lr_%.4f-COS_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d" % (
             self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
             self.args.momentum, self.args.decay, self.args.seed)
        if "cos" in mode:
            self.args.save_path = self.args.save_path + "-T_%.2f" % self.args.temperature
        if "ft" in self.args.new_mode:
            self.args.save_path = self.args.save_path + "-ftLR_%.3f-ftEpoch_%d" % (
             self.args.lr_new, self.args.epochs_new)
        if self.args.debug:
            self.args.save_path = os.path.join("debug", self.args.save_path)
        self.args.save_path = os.path.join("checkpoint/Ours_Baseline", self.args.save_path)
        ensure_path(self.args.save_path)

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = "%s/" % self.args.project
            self.args.save_log_path = self.args.save_log_path + "%s" % self.args.dataset
            if "avg" in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + "_prototype_" + self.args.model_dir.split("/")[-2][:7] + "/"
            if "ft" in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + "_WaRP_" + "lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/" % (
                 self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join("acc_logs", self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split("/")[-2] + ".csv"

