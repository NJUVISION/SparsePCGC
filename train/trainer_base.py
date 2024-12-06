import os, sys
rootdir = os.path.split(__file__)[0]
# sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
# sys.path.append(rootdir)
import os, logging
from tensorboardX import SummaryWriter
import numpy as np
import torch
import time
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)


class Trainer():
    def __init__(self, model, prefix='tp', init_ckpt='', rootdir=None, pretrained_modules=None, 
                frozen_modules=None, lr=0.0001, check_time=10, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.prefix = prefix
        if rootdir is None: rootdir = os.path.split(os.path.split(__file__)[0])[0]
        self.ckptdir = os.path.join(rootdir, 'ckpts', prefix)
        os.makedirs(self.ckptdir, exist_ok=True)
        self.logger = self.getlogger(logdir=self.ckptdir)
        self.writer = SummaryWriter(log_dir=self.ckptdir)# tensorboard
        # self.MSELoss = torch.nn.MSELoss().to(device)
        self.pretrained_modules = pretrained_modules
        self.frozen_modules = frozen_modules
        _ = self.load_state_dict(init_ckpt)# load ckpt
        self.lr = lr
        self.epoch = 0
        self.check_time = check_time# minute
        self.record_set = {}

    def getlogger(self, logdir=None):
        logger = logging.getLogger(__name__)
        logger.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if logdir:
            file_handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def load_state_dict(self, init_ckpt):
        if init_ckpt=='':
            self.logger.info('Random initialization.')
            pretrained_dict_keys = {}
        else:
            ckpt = torch.load(init_ckpt)
            model_dict = self.model.state_dict()
            if self.pretrained_modules:
                pretrained_dict = {k:v for k,v in ckpt['model'].items() \
                    if k in model_dict and k.split('.')[0] in self.pretrained_modules}
            else:
                pretrained_dict = {k:v for k,v in ckpt['model'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)
            self.logger.info('Load checkpoint from ' + init_ckpt)
            pretrained_dict_keys = [k.split('.')[0] for k in pretrained_dict.keys()]
            pretrained_dict_keys = np.unique(pretrained_dict_keys).tolist()
            self.logger.info('Load pretained modules:' + str(pretrained_dict_keys))
            # self.model.model.load_state_dict(ckpt['model'])

        return pretrained_dict_keys

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 
                    os.path.join(self.ckptdir, 'epoch_last.pth'))
        # torch.save({'model': self.model.state_dict()}, 
        #             os.path.join(self.ckptdir, str(self.epoch)+'.pth'))

        return

    def set_optimizer(self):
        self.logger.info('='*5+' set_optimizer'+'='*5)
        params_lr_list = []
        for module_name in self.model._modules.keys():
            if self.frozen_modules and module_name in self.frozen_modules: 
                for v in self.model._modules[module_name].parameters():v.requires_grad = False
                self.logger.info('frozen: '+module_name+'\tlr:0')
            else: 
                lr = self.lr
                params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':lr})
                self.logger.info('optimize: '+module_name+'\tlr:'+str(lr))
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)
        
        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10 + main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items():
            self.record_set[k]=np.mean(np.array(v), axis=0).squeeze()
        for k, v in self.record_set.items():
            self.logger.info(k+': '+str(np.round(v, 4)))
            if len(v.shape) > 1: self.logger.info(k+' sum: '+str(np.round(v.sum(axis=-1), 3).tolist()))
        # tensorboard            
        tag_scalar_dict = {}
        for k, v in self.record_set.items(): tag_scalar_dict[k] = np.sum(v)
        self.writer.add_scalars(main_tag=main_tag, 
                                tag_scalar_dict=tag_scalar_dict, 
                                global_step=global_step)
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  

        return

    def clip_gradient(self, max_norm=None, clip_value=None, DBG=True):
        # for p in self.model.parameters():
        #     if p.grad is None: print('DBG!!! None')
        #     else: print('DBG!!!', p.grad.data.shape, p.grad.data.abs().max().item())
        if DBG: max_grad = max(p.grad.data.abs().max() for p in self.model.parameters())
        # clip by norm
        if max_norm is not None: 
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm) 
            if DBG:
                total_norm2 = 0
                for p in self.model.parameters(): total_norm2+=p.grad.data.norm(2)**2
                total_norm2 = total_norm2 ** (1./2)
                if round(total_norm2.item(),4)<round(total_norm.item(),4) and DBG:
                    print('total_norm:', round(total_norm.item(),4), ' -> ', round(total_norm2.item(),4), '\tthreshold:', max_norm)
        # clip by value
        if clip_value is not None: 
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value)
        if DBG:
            max_grad2 = max(p.grad.data.abs().max() for p in self.model.parameters())
            if max_grad2.item()<max_grad.item() and DBG:
                print('max_grad:', round(max_grad.item(),4), ' -> ', round(max_grad2.item(),4), '\tthreshold:', clip_value)
        
        return 

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch) + '\t' + self.prefix)
        self.optimizer = self.set_optimizer()# optimizer
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        self.logger.info('Training Files length:' + str(len(dataloader)))
        start_time = time.time()
        for batch_step, data in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            try:
                loss = self.forward(data, training=True)# forward
                # with torch.autograd.detect_anomaly(): 
                loss.backward()# backward
                # self.clip_gradient(max_norm=4)# 1
                self.optimizer.step()# optimize
            except (RuntimeError, ValueError, MemoryError) as e:
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                time.sleep(2)
                self.logger.info('!'*20+' except: '+str(e)+'!'*20)
                continue
            # check & record
            if (time.time() - start_time) > self.check_time*60:
                self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
                self.save_model()
                start_time = time.time()
                print("memoey!:\t", round(torch.cuda.max_memory_allocated()/1024**3,3),'GB')
            torch.cuda.empty_cache()
        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        self.save_model()
        self.epoch += 1

        return

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        self.logger.info('Testing Files length:' + str(len(dataloader)) + '\t' + self.prefix)
        for _, data in enumerate(tqdm(dataloader)):
            _ = self.forward(data, training=False)# forward
            torch.cuda.empty_cache()
        print("memoey!:\t", round(torch.cuda.max_memory_allocated()/1024**3,3),'GB')
        self.record(main_tag=main_tag, global_step=self.epoch)

        return

    def forward(self):
        return