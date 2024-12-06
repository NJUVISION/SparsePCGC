import os, sys
rootdir = os.path.split(__file__)[0]
sys.path.append(rootdir)
rootdir = os.path.split(rootdir)[0]
sys.path.append(rootdir)
import torch
import MinkowskiEngine as ME
import numpy as np
import os, glob, argparse, random
from data_utils.data_loader import PCDataset, make_data_loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from trainer_base import Trainer
class OffsetTrainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MSELoss = torch.nn.MSELoss().to(device)
        self.logger.info('\n'.join(['%s:\t%s' % item for item in self.__dict__.items() if item[0]!='model']))
        self.logger.info(args.dataset)
        self.logger.info(args.dataset_test)
        para = sum([np.prod(list(p.size())) for p in self.model.parameters()])
        self.logger.info('params: '+str(para)+'\tmodel size: '+str(round(para*4/1024))+' KB')
        # self.logger.info(self.model)

    def forward(self, data, training):
        coords, feats = data
        # data
        x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
        if (x.shape[0] < args.min_points or x.shape[0] > args.max_points) and training==True:
            self.logger.info('num_points:\t' + str(x.shape[0]))
            return torch.autograd.Variable(torch.Tensor([0]), requires_grad=True)
        # forward
        out_set_list = self.model(x)
        loss = 0
        mse_list = []
        for _, out_set in enumerate(out_set_list):
            mse = self.MSELoss(out_set['out'].F, out_set['ground_truth'].F).to(device)
            loss += mse
            mse_list.append(mse.item())
        # record
        record_set = {}
        with torch.no_grad():
            record_set.update({'loss':loss.item(), 'mse_list':mse_list})
        # collect record
        for k, v in record_set.items():
            if k not in self.record_set: self.record_set[k]=[]
            self.record_set[k].append(v)

        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # model config
    parser.add_argument("--model", type=str, default='')  
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--block_layers", type=int, default=3)

    # dataset config
    parser.add_argument("--dataset",type=str, default='../dataset/KITTI/pc_q1mm/train_part/')
    parser.add_argument("--dataset_num", type=int, default=5000)
    parser.add_argument("--dataset_test",type=str, default='../dataset/KITTI/pc_q1mm/test100/')
    parser.add_argument("--test_num", type=int, default=int(100))
    parser.add_argument("--voxel_size", type=float, default=1)
    parser.add_argument("--max_points", type=int, default=int(1000000))
    parser.add_argument("--min_points", type=int, default=int(1000))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--augment", action="store_true", help="test or not.")# random_quantize
    parser.add_argument("--posQuantscale_list", type=int, nargs='+')
    # training config
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument('--pretrained_modules', nargs='+')
    parser.add_argument('--frozen_modules', nargs='+')
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--check_time", type=float, default=10, help='(min).')
    # global config
    parser.add_argument("--only_test", action="store_true", help="test or not.")
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--prefix", type=str, default='tp', help="prefix of checkpoints/logger, etc.")
    args = parser.parse_args()

    # model
    from models.model_offset import OffsetModel
    model = OffsetModel(channels=args.channels, kernel_size=args.kernel_size, block_layers=args.block_layers, 
                        posQuantscale_list=args.posQuantscale_list).to(device)
    print('model', model)
    # trainer
    trainer = OffsetTrainer(model=model, lr=args.lr, init_ckpt=args.init_ckpt, 
                            pretrained_modules=args.pretrained_modules, frozen_modules=args.frozen_modules, 
                            prefix=args.prefix, check_time=args.check_time, device=device)
    # data
    all_filedirs = sorted(glob.glob(os.path.join(args.dataset,'**', f'*.*'), recursive=True))
    all_filedirs = [f for f in all_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
    print('all file length:\t', len(all_filedirs))
    # val dataset
    val_filedirs = all_filedirs[::len(all_filedirs)//100]
    val_dataset = PCDataset(val_filedirs, voxel_size=args.voxel_size)
    val_dataloader = make_data_loader(dataset=val_dataset, batch_size=1, shuffle=False, repeat=False)
    print('validate file length:\t', len(val_filedirs))
    # test dataset
    test_filedirs = sorted(glob.glob(os.path.join(args.dataset_test,'**', f'*.*'), recursive=True))
    test_filedirs = [f for f in test_filedirs if f.endswith('h5') or f.endswith('ply') or f.endswith('bin')]
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
        if epoch>0  and epoch%2==0:
            args.lr =  max(args.lr/2, args.lr_min)
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
        trainer.test(val_dataloader, 'Val')