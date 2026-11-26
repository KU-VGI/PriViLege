from .base import Trainer
import os.path as osp
import torch.nn as nn
from .parallel import DataParallelModel, DataParallelCriterion
import copy
from copy import deepcopy
import pandas as pd
from os.path import exists as is_exists
import json
import random
import numpy as np

from .helper import *
from utils import *
from dataloader.data_utils import *
from models.switch_module import switch_module
from dataloader.data_manager import DataManager

class ViT_FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.set_log_path()

        self.args = set_up_datasets(self.args)
        if self.args.pret_clip:
            # Restored from the authors' compiled trainer: PriViLege on CLIP (paper Figure 3)
            from .ViT_Network_CLIP import ViT_MYNET_CLIP
            self.model = ViT_MYNET_CLIP(self.args, mode=self.args.base_mode)
        else:
            self.model = ViT_MYNET(self.args, mode=self.args.base_mode)

        if self.args.LT:
            print("Tuning Layer!!")
            if self.args.pret_clip:
                for p in self.model.text_encoder.parameters():
                    p.requires_grad=False
                
                for p in self.model.encoder.parameters():
                    p.requires_grad=False
                    
                num_layer = [0,1]
                for idx, block in enumerate(self.model.encoder.transformer.resblocks):
                    if idx in num_layer:
                        for p in block.parameters():
                            p.requires_grad=True
                
                for p in self.model.encoder.transformer.key_comp.parameters():
                    p.requires_grad=True
                for p in self.model.encoder.transformer.val_comp.parameters():
                    p.requires_grad=True
                
                self.model.encoder.prompt.requires_grad=True
                self.model.encoder.expert_prompt.requires_grad=True
            
            else:
                #! 마지막 2 Layer만 Freeze --> Prototype Classifier의 Bias 문제 해결위해?
                for p in self.model.encoder.parameters():
                    p.requires_grad=False
                
                num_layer = [l for l in range(args.taskblock)] 
                for idx, block in enumerate(self.model.encoder.blocks):
                    if idx in num_layer:
                        for p in block.parameters():
                            p.requires_grad=True
        elif self.args.scratch:
            print("Scratch Model!!")
        elif self.args.ft:
            for p in self.model.parameters():
                p.requires_grad=False
                
            num_layer = [l for l in range(args.taskblock)]
            for idx, block in enumerate(self.model.encoder.blocks):
                if idx in num_layer:
                    for p in block.parameters():
                        p.requires_grad=True
            if self.args.plain_ft:   # plain fine-tuning: the classifier is trained with the tuned blocks
                for p in self.model.fc.parameters():
                    p.requires_grad=True
        elif self.args.lp:
            pass
        else:
            for p in self.model.encoder.parameters():
                p.requires_grad=False
            print("No Tuning Layer!!")
        
        if self.args.pret_clip:
            self.val_model = ViT_MYNET_CLIP(self.args, mode=self.args.base_mode)
        else:
            self.val_model = ViT_MYNET(self.args, mode=self.args.base_mode)

        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        #* val_model is never used. It is still built above, so the random-number sequence (and so every
        #* seed's data order) matches the authors' code, but it is not kept: on a 16 GB T4 it would hold
        #* another full encoder in GPU memory.
        del self.val_model

        self.word_info = {}
        self.word_info["embed"] = None
        self.word_info["cur_embed"] = None
        self.word_info["label_text"] = np.array([])
        
        self.query_info={}
        self.query_info["proto"] = None
        
        self.loss_curve={}
        self.loss_curve['ACC'] = []
        self.loss_curve['CE_loss'] = []
        self.loss_curve['ED_loss']=[]
        self.loss_curve['ED_ce']=[]
        self.loss_curve['ED_kl']=[]
        self.loss_curve['ED_loss']=[]
        self.loss_curve['SKD_loss']=[]
        self.loss_curve['SKD_kd']=[]
        self.loss_curve['SKD_ce']=[]
        self.loss_curve['total_loss']=[]
        self.loss_curve['grad_list'] = []
        
        self.loss_curve['attn_score']=[]
        
        #* Bert_model for ViT-B
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.Bert_model = BertModel.from_pretrained("bert-base-cased")
        self.Bert_model = nn.DataParallel(self.Bert_model, list(range(self.args.num_gpu)))
        self.Bert_model = self.Bert_model.cuda()
        
        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']

        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())
    
    
        print("#"*50)
        trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        self.init_params = sum(param.numel() for param in self.model.parameters())
        print('total parameters:',self.init_params)
        print('trainable parameters:',trainable_params)
        print("#"*50)

    def get_optimizer_base(self):
        #! Original
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr_base,)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def get_standard_dataloader(self, session, data_manager):
        # import data_manager
        batchsize=128
        num_cls=10
        train_dataset = data_manager.get_dataset(
            np.arange(session*num_cls, (session+1)*num_cls),
            source="train",
            mode="train",
            # appendent=self._get_memory(),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batchsize, shuffle=True, num_workers=4
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, (session+1)*num_cls), source="test", mode="test"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batchsize, shuffle=False, num_workers=4
        )
        return data_manager.idata.train_set, train_loader, test_loader


    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        columns = ['num_session', 'acc', 'base_acc', 'new_acc', 'base_acc_given_new', 'new_acc_given_base']
        acc_df = pd.DataFrame(columns=columns)

        #* Resume (only with -ckpt_dir): continue after the last saved base epoch or session
        resume = None
        start_session, start_epoch = args.start_session, 0
        if self._resume_enabled() and os.path.exists(self._resume_path()):
            resume = self._load_resume()   # None if the saved state held NaN/inf weights: start fresh
        if resume is not None:
            result_list = [args] + resume['result_list']
            if resume['phase'] == 'base':
                start_session, start_epoch = 0, resume['epoch'] + 1
                print('[resume] continuing the base session at epoch {}/{}'.format(start_epoch + 1, args.epochs_base))
            else:
                start_session = resume['session'] + 1
                print('[resume] continuing at session {}/{}'.format(start_session, args.sessions - 1))
            if resume['phase'] == 'session':
                self._restore_rng(resume['rng'])
        print("[Start Session: {}] [Sessions: {}]".format(start_session, args.sessions))

        for session in range(start_session, args.sessions):
            
            train_set, trainloader, testloader = self.get_dataloader(session)
            print(f"Session: {session} Data Config")
            print(len(train_set.targets))
            if session > 0:
                print("Freeze parameters of the encoder.. ")
                if args.pret_clip:
                    for idx, block in enumerate(self.model.module.encoder.transformer.resblocks):
                        for p in block.parameters():
                            p.requires_grad=False
                    for p in self.model.module.encoder.transformer.key_comp.parameters():
                        p.requires_grad = False
                    for p in self.model.module.encoder.transformer.val_comp.parameters():   # restored from the authors' compiled trainer
                        p.requires_grad = False
                    self.model.module.encoder.expert_prompt.requires_grad=False
                elif args.ft:
                    for p in self.model.parameters():
                        p.requires_grad=True
                elif args.lp:
                    for p in self.model.module.parameters():
                        p.requires_grad=False
                    
                    for p in self.model.module.fc.parameters():
                        p.requires_grad=True
                else:
                    for p in self.model.module.encoder.parameters():
                        p.requires_grad=False
                    self.model.module.expert_prompt.requires_grad = False
                    #* Pointwise_Compressor
                    for p in self.model.module.global_comp.parameters():
                        p.requires_grad = False
                    
                    for p in self.model.module.local_comps.parameters():
                        p.requires_grad = False
            #todo ===============================================
            if session == 0:  # load base class train img label
                
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                if resume is not None:
                    #* Word embeddings and base prototypes were restored with the resume state
                    optimizer.load_state_dict(resume['optimizer'])
                    scheduler.load_state_dict(resume['scheduler'])
                    self.Bert_model = self.Bert_model.cpu()
                else:
                    #todo Build Word Embedding..
                    print("Build Word Information..")
                    if args.pret_clip:
                        build_label_embedding(train_set, session, self.model.module, self.tokenizer, self.word_info, args)
                    else:
                        build_label_embedding(train_set, session, self.Bert_model, self.tokenizer, self.word_info, args)
                    self.Bert_model = self.Bert_model.cpu()
                    print("Total Word vector info:", self.word_info["embed"].shape)
                    print("Current Word vector info:", self.word_info["cur_embed"].shape)
                    print("Current Word label info:", self.word_info["label_text"].shape)
                    print()
                    # if not self.args.pret_clip:
                    print("Build Base query prototype Information..")
                    build_base_proto(trainloader, self.model, self.query_info, args)
                print("Base Proto vector info:", self.query_info["proto"].shape)
                # print("Base Proto Matrix info:", self.query_info["rel_matrix"].shape)
                print("[Base Session Training]")
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
                if resume is not None:
                    self._restore_rng(resume['rng'])
                for epoch in range(start_epoch, args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, self.word_info, self.query_info, np.unique(train_set.targets), args, self.loss_curve)
                    tsl, tsa, logs = test(self.model, testloader, epoch, args, session,self.word_info)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                            '\nstill need around %.2f mins to finish this session' % (
                                    (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()
                    self._save_resume('base', session, epoch, result_list, optimizer, scheduler)

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_last_epoch.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                
                self.best_model_dict = deepcopy(self.model.state_dict())
                #*=======================================================================================
                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    if not args.pret_clip:
                        self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                        best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc_replace_head.pth')
                        print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, logs = test(self.model, testloader, 0, args, session,self.word_info)
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))
                self._save_resume('session', session, None, result_list)

            else:  # incremental learning sessions
                print("Incremental session: [%d]" % session)
                print("#"*50)
                trainable_params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
                print('[Session {}] Trainable parameters: {}'.format(session,trainable_params))
                print("#"*50)
                #todo Build Word Embedding..
                print("Build Word Information..")
                self.Bert_model = self.Bert_model.cuda()
                if args.pret_clip:
                    build_label_embedding(train_set, session, self.model.module, self.tokenizer, self.word_info, args)
                else:
                    build_label_embedding(train_set, session, self.Bert_model, self.tokenizer, self.word_info, args)
                self.Bert_model = self.Bert_model.cpu()
                print("Total Word vector info:", self.word_info["embed"].shape)
                print("Current Word vector info:", self.word_info["cur_embed"].shape)
                print("Current Word label info:", self.word_info["label_text"].shape)
                print()
                self.model.module.update_seen_classes(np.unique(train_set.targets))

                self.model.module.mode = self.args.new_mode
                self.model.train()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.train_inc(trainloader, self.args.epochs_new, session, np.unique(train_set.targets), self.word_info, self.query_info)
                self.model.eval()
                self.model.module.mode = 'avg_cos'
                tsl, tsa, logs = test(self.model, testloader, 0, args, session,self.word_info)
                acc_df = pd.concat([acc_df, pd.DataFrame([logs])], ignore_index=True)
                
                print("Build Vision ProtoType")

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))
                self._save_resume('session', session, None, result_list)

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)
        
        # end_params = 0.
        end_params = sum(param.numel() for param in self.model.module.parameters())
        print('[Begin] Total parameters: {}'.format(self.init_params))
        print('[END] Total parameters: {}'.format(end_params))
        
    #* ------------------------------------------------------------------
    #* Resume support (-ckpt_dir). A state file is saved after every base
    #* epoch and every session; a restarted run continues from it.
    #* ------------------------------------------------------------------
    RESUME_ARGS = ['dataset', 'seed', 'epochs_base', 'epochs_new', 'lr_base', 'lr_new', 'batch_size_base',
                   'schedule', 'base_mode', 'new_mode', 'ED', 'ED_hp', 'SKD', 'LT', 'taskblock', 'comp_out',
                   'prefix', 'vit', 'pret_clip']

    def _resume_enabled(self):
        return bool(getattr(self.args, 'ckpt_dir', None))

    def _resume_path(self):
        return os.path.join(self.args.ckpt_dir, 'resume_state.pt')

    def _resume_fingerprint(self):
        return {k: getattr(self.args, k, None) for k in self.RESUME_ARGS}

    def _model_state_for_resume(self):
        #* With -LT, encoder weights outside the tuned blocks stay at their pre-trained
        #* values and timm reloads them on restart, so they are not saved.
        state = self.model.state_dict()
        if not self.args.LT or self.args.pret_clip:
            return state
        keep = tuple('module.encoder.blocks.%d.' % i for i in range(self.args.taskblock)) + ('module.encoder.head.',)
        return {k: v for k, v in state.items() if not k.startswith('module.encoder.') or k.startswith(keep)}

    @staticmethod
    def _nonfinite_keys(state_dict):
        return [k for k, v in state_dict.items()
                if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]

    def _save_resume(self, phase, session, epoch, result_list, optimizer=None, scheduler=None):
        if not self._resume_enabled():
            return
        model_state = self._model_state_for_resume()
        bad = self._nonfinite_keys(model_state)
        if bad:   # never replace the last good resume state with broken weights
            raise FloatingPointError('Not saving the resume state: NaN/inf weights in {} (e.g. {}). The previous resume '
                                     'state is kept.'.format(len(bad), bad[:3]))
        state = {
            'phase': phase, 'session': session, 'epoch': epoch,
            'fingerprint': self._resume_fingerprint(),
            'model': model_state,
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'trlog': self.trlog,
            'word_info': self.word_info,
            'query_info': self.query_info,
            'seen_classes': self.model.module.seen_classes,
            'result_list': result_list[1:],
            'rng': {'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'numpy': np.random.get_state(),
                    'python': random.getstate()},
        }
        os.makedirs(self.args.ckpt_dir, exist_ok=True)
        path = self._resume_path()
        torch.save(state, path + '.tmp')
        os.replace(path + '.tmp', path)
        info = {'phase': phase, 'session': session, 'epoch': epoch, 'sessions': self.args.sessions,
                'epochs_base': self.args.epochs_base, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(os.path.join(self.args.ckpt_dir, 'resume_state.json'), 'w') as f:
            json.dump(info, f)
        print('[resume] state saved: phase={} session={} epoch={}'.format(phase, session, epoch))

    def _load_resume(self):
        path = self._resume_path()
        try:
            state = torch.load(path, map_location='cpu', weights_only=False)
        except TypeError:  # torch < 1.13 has no weights_only
            state = torch.load(path, map_location='cpu')
        if state['fingerprint'] != self._resume_fingerprint():
            raise ValueError('Resume state in {} was saved with different arguments.\nsaved: {}\nnow:   {}\n'
                             'Delete that folder or use another -ckpt_dir.'.format(
                                 self.args.ckpt_dir, state['fingerprint'], self._resume_fingerprint()))
        bad = self._nonfinite_keys(state['model'])
        if bad:   # e.g. saved by the code from before the L_ED NaN fix: unusable, so set it aside and start fresh
            os.replace(path, path + '.nan')
            print('[resume] {} has NaN/inf weights ({} tensors, e.g. {}): moved it to {}.nan and starting this run '
                  'from scratch.'.format(path, len(bad), bad[:3], path))
            return None
        missing, unexpected = self.model.load_state_dict(state['model'], strict=False)
        not_encoder = [k for k in missing if not k.startswith('module.encoder.')]
        if unexpected or not_encoder:
            raise RuntimeError('Resume state does not match the model: unexpected={} missing={}'.format(
                unexpected, not_encoder))
        self.trlog = state['trlog']
        self.word_info = state['word_info']
        self.query_info = state['query_info']
        self.model.module.seen_classes = state['seen_classes']
        print('[resume] loaded {} (phase={} session={} epoch={})'.format(
            path, state['phase'], state['session'], state['epoch']))
        return state

    def _restore_rng(self, rng):
        torch.set_rng_state(rng['torch'])
        if torch.cuda.is_available() and rng['cuda']:
            torch.cuda.set_rng_state_all(rng['cuda'])
        np.random.set_state(rng['numpy'])
        random.setstate(rng['python'])

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        if self.args.vit:
            self.args.save_path = self.args.save_path + '%s/' % (self.args.project+'_ViT_Ours')
        else:
            self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        else:
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-COS_%d-Gam_%.2f-Bs_%d-Mom_%.2f-Wd_%.5f-seed_%d' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum, self.args.decay, self.args.seed)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join(f'checkpoint/{self.args.out}', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def set_log_path(self):
        if self.args.model_dir is not None:
            self.args.save_log_path = '%s/' % self.args.project
            self.args.save_log_path = self.args.save_log_path + '%s' % self.args.dataset
            if 'avg' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_prototype_' + self.args.model_dir.split('/')[-2][:7] + '/'
            if 'ft' in self.args.new_mode:
                self.args.save_log_path = self.args.save_log_path + '_WaRP_' + 'lr_new_%.3f-epochs_new_%d-keep_frac_%.2f/' % (
                    self.args.lr_new, self.args.epochs_new, self.args.fraction_to_keep)
            self.args.save_log_path = os.path.join('acc_logs', self.args.save_log_path)
            ensure_path(self.args.save_log_path)
            self.args.save_log_path = self.args.save_log_path + self.args.model_dir.split('/')[-2] + '.csv'

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print ('Error: Creating directory. ' +  directory)