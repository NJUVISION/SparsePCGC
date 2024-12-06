import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import MinkowskiEngine as ME
import numpy as np
import os, glob, argparse, time, random, logging
from tqdm import tqdm
from data_utils.quantize import random_quantize
from data_utils.data_loader import PCDataset, make_data_loader
from models.loss import get_bce, get_bits
from tensorboardX import SummaryWriter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)


from trainer_base import Trainer
class PCCTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger.info('\n'.join(['%s:\t%s' % item for item in self.__dict__.items() if item[0]!='model']))
        self.logger.info(args.dataset)
        self.logger.info(args.dataset_test)
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        self.logger.info('params: '+str(para)+'\tmodel size: '+str(round(para*4/1024))+' KB')
        # self.logger.info(self.model)

    def forward(self, data, training):
        coords, feats = data
        # print('DBG!!!!', len(coords))
        # data
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        if (x.shape[0] < args.min_points or x.shape[0] > args.max_points) and training==True:
            self.logger.info('num_points:\t' + str(x.shape[0]))
            return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True)
        # forward
        out_set_list = self.model(x)
        loss = 0
        bce_matrix, bpp_list = [], []
        for _, out_set in enumerate(out_set_list):
            # current scale
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                # current stage
                curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())
            if 'likelihood' in out_set: 
                bpp = get_bits(out_set['likelihood'])/float(x.__len__())
            else: 
                bpp = torch.tensor([0]).to(bce.device)
            # print('DBG!!!', args.alpha, args.beta)
            curr_loss = args.alpha * bce +args.beta * bpp
            loss += curr_loss
            bce_matrix.append(bce_list)
            bpp_list.append(bpp.item())
        # record
        record_set = {}
        with torch.no_grad():
            record_set.update({'loss':np.sum(bce_matrix)+np.sum(bpp_list)})
            record_set.update({'bce_matrix':bce_matrix})
            record_set.update({'bpp_list':bpp_list})
        # collect record
        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)

        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model config
    parser.add_argument("--enc_type", type=str, default='pooling')
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    # parser.add_argument("--block_type", type=str, default='inception')
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--block_layers", type=int, default=3)
    parser.add_argument("--scale", type=int, default=4)
    # dataset config
    parser.add_argument("--dataset", type=str, default='../dataset/KITTI/pc_q1mm/train_part/')
    parser.add_argument("--dataset_num", type=int, default=int(5000))
    parser.add_argument("--test_num", type=int, default=int(100))
    parser.add_argument("--dataset_test", type=str, default='../dataset/KITTI/pc_q1mm/test100/')
    parser.add_argument("--voxel_size", type=float, default=1)
    parser.add_argument("--max_points", type=int, default=int(2000000))
    parser.add_argument("--min_points", type=int, default=int(1000))
    parser.add_argument("--augment", action="store_true", help="test or not.")# random_quantize
    parser.add_argument("--batch_size", type=int, default=1)
    # training config
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument('--pretrained_modules', nargs='+')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--check_time", type=float, default=10, help='frequency for recording state (min).')
    # global config
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--only_test", action="store_true", help="test or not.")
    parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=0., help="weights for bit rate.")
    args = parser.parse_args()

    # model
    from models.model import PCCModel
    model = PCCModel(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers,
                    stage=args.stage, scale=args.scale,  enc_type=args.enc_type).to(device)
    print(model)
    # trainer
    trainer = PCCTrainer(model=model, lr=args.lr, init_ckpt=args.init_ckpt, 
                        pretrained_modules=args.pretrained_modules, frozen_modules=args.frozen_modules, 
                        prefix=args.prefix, check_time=args.check_time, device=device)
    # data
    all_filedirs = sorted(glob.glob(os.path.join(args.dataset,'**', f'*.*'), recursive=True))
    all_filedirs = [f for f in all_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]

    print('file length:\t', len(all_filedirs))
    # val dataset
    val_filedirs = all_filedirs[::len(all_filedirs)//100]
    val_dataset = PCDataset(val_filedirs, voxel_size=args.voxel_size)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)
    print('validate file length:\t', len(val_filedirs))
    # test dataset
    test_filedirs = sorted(glob.glob(os.path.join(args.dataset_test,'**', f'*.*'), recursive=True))
    test_filedirs = [f for f in test_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    if len(test_filedirs)>100: test_filedirs = test_filedirs[::len(test_filedirs)//100]
    test_dataset = PCDataset(test_filedirs, voxel_size=args.voxel_size)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=1, shuffle=False, repeat=False)
    print('test file length:\t', len(test_filedirs))

    # training
    for epoch in range(0, args.epoch):
        # only test
        if args.only_test: 
            trainer.test(test_dataloader, 'Test')
            trainer.test(val_dataloader, 'Val')
            break
        # training dataset
        filedirs = random.sample(all_filedirs, min(len(all_filedirs), args.dataset_num))
        print('file length:\t', len(filedirs))
        train_dataset = PCDataset(filedirs, voxel_size=args.voxel_size, augment=args.augment)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, 
                                            shuffle=True, repeat=False)
        # train
        if epoch>0 and epoch%2==0: 
            args.lr = max(args.lr/2, args.lr_min)
        # update weight
        # if args.enc_type[:2]=='ae':
        #     if epoch<4: args.beta=0
        #     elif epoch<8: args.beta=0.1
        #     elif epoch<12: args.beta=0.2
        #     elif epoch<16: args.beta=0.5
        #     else: args.beta=1
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Val')