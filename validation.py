# -*- coding:utf-8 -*-

import argparse

from torch.utils.data import DataLoader
from dataset.dataset_sig17 import  SIG17_Validation_Dataset


from utils.utils import *

from models.pgn import  pgn
def get_args():
    parser = argparse.ArgumentParser(description='pgn',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset_dir", type=str, default='./data',
                        help='dataset directory'),
    parser.add_argument('--logdir', type=str, default='./checkpoints',
                        help='target log directory')
    parser.add_argument('--num_workers', type=int, default=0, metavar='N',
                        help='number of workers to fetch data (default: 8)')
    parser.add_argument('--resume', type=str, default='/best_checkpoint.pth',
                        help='load model from a .pth file')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    return parser.parse_args()
def validation(args, model, device, val_loader):
    model.eval()
    val_psnr = AverageMeter()
    val_mu_psnr = AverageMeter()
    val_ssim = AverageMeter()
    val_mu_ssim = AverageMeter()
    val_loss = AverageMeter()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):


            batch_ldr0, batch_ldr1, batch_ldr2 = batch_data['input0'].to(device), batch_data['input1'].to(device), batch_data['input2'].to(device)
            edge1=batch_data['edge1'].to(device)
            edge2=batch_data['edge2'].to(device)
            edge3=batch_data['edge3'].to(device)
            label = batch_data['label'].to(device)
            pred = model(batch_ldr0, batch_ldr1, batch_ldr2,edge1,edge2,edge3)
            psnr = batch_psnr(pred, label, 1.0)
            mu_psnr = batch_psnr_mu(pred, label, 1.0)
            ssim=batch_ssim(pred, label)
            mu_ssim=batch_ssim_mu(pred, label)
            val_psnr.update(psnr.item())
            val_mu_psnr.update(mu_psnr.item())
            val_ssim.update(ssim.item())
            val_mu_ssim.update(mu_ssim.item())
            pred=pred.cpu().numpy().squeeze(0).transpose(1,2,0)[..., ::-1]
            args.save_results='./results'+'/'
            if args.save_results:
                  if not os.path.exists(args.save_results):
                        os.makedirs(args.save_results)
                  save_dir=os.path.join(args.save_results, '{}_pred.hdr'.format(batch_idx))
                  save_hdr(save_dir, pred)
            print(batch_idx)
            print(psnr)
            print(mu_psnr)
            print(ssim)
            print(mu_ssim)

    print('Validation set: Average Loss: {:.4f}'.format(val_loss.avg))
    print('Validation set: Average PSNR: {:.4f}, mu_law: {:.4f}'.format(val_psnr.avg, val_mu_psnr.avg))
    print('Validation set: Average SSIM: {:.4f}, mu_law: {:.4f}'.format(val_ssim.avg, val_mu_ssim.avg))




def main():
    # settings
    args = get_args()
    args.resume=args.logdir+args.resume
    # cuda and devices
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = pgn()
    model.to(device)
    model = nn.DataParallel(model)
    if args.resume:
        if os.path.isfile(args.resume):
            print("===> Loading checkpoint from: {}".format(args.resume))
            checkpoint = torch.load(args.resume,map_location='cuda:0')
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("===> No checkpoint is founded at {}.".format(args.resume))
    val_dataset = SIG17_Validation_Dataset(root_dir=args.dataset_dir, is_training=False, crop=False, crop_size=512)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    validation(args, model, device, val_loader)


if __name__ == '__main__':
    main()
