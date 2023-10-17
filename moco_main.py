from argparse import ArgumentParser
import gc
import json
import math
import shutil
import torch
import torchvision.transforms as transforms
import os
from tqdm import tqdm
import wandb
import yaml

from moco.datasets.road_dataset import build_moco_road_dataset
from moco.loader import GaussianBlur, Solarize
from moco.model import MoCo

os.environ['WANDB_API_KEY'] = '94ab84459aa1b42734e2980087f053e645c271e7'

parser = ArgumentParser()
parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. ',
                    required=True)
parser.add_argument('--resume', default=None,
                    help='checkpoint of the last epoch of the model')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def main(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(args.exp_name)
    config = dict2obj(config)
    config.log.exp_name = args.exp_name

    exp_path = os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (
        config.log.exp_name, config.DATA.SEED))
    if os.path.exists(exp_path):
        print('ERROR: Experiment folder exist, please change exp name in config file')
    else:
        try:
            os.makedirs(exp_path)
            shutil.copyfile(args.config, os.path.join(exp_path, "config.yaml"))
        except:
            pass

    config.DATA.MIXED = False

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    # Load model
    model = MoCo(config, device, dim=256, mlp_dim=4096, T=0.2)
    model.to(device)

    # Load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY))

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="relationformer_moco",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": float(config.TRAIN.LR),
            "epochs": config.TRAIN.EPOCHS,
            "batch_size": config.DATA.BATCH_SIZE,
            "accumulated batch size": config.DATA.TARGET_BATCH_SIZE,
        }
    )

    augmentation1 = [
        transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    augmentation2 = [
        transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
        transforms.RandomApply([Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]

    # Create dataset and dataloader
    train_ds, val_ds = build_moco_road_dataset(config, transforms.Compose(augmentation1), transforms.Compose(augmentation2), max_samples=config.DATA.NUM_SOURCE_SAMPLES)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.DATA.BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, drop_last=True)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            # Map model to be loaded to specified single gpu.
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    accum_iter = config.DATA.TARGET_BATCH_SIZE // config.DATA.BATCH_SIZE

    # Training Loop
    for epoch in range(config.TRAIN.EPOCHS):
        print("epoch ", epoch)
        train(train_loader, model, optimizer, epoch, args, device, accum_iter, config)

        #validate(val_loader, model, epoch, args)

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, is_best=False, filename=f'trained_weights/runs/{args.exp_name}/checkpoint_%04d.pth.tar' % epoch)

        

def train(train_loader, model, optimizer, epoch, args, device, accum_iter, config):
    # switch to train mode
    model.train()

    # Accumulate qs and ks for accum_iter to compute loss
    q1s = []
    q2s = []
    k1s = []
    k2s = []

    for iteration, (x1, x2) in enumerate(tqdm(train_loader)):
        moco_m = 0.99

        # Move data to device
        x1 = x1.to(device)
        x2 = x2.to(device)

        # Forward pass
        q1, q2, k1, k2 = model(x1, x2, moco_m)
        q1s.append(q1)
        q2s.append(q2)
        k1s.append(k1)
        k2s.append(k2)

        if ((iteration + 1) % accum_iter == 0) or (iteration + 1 == len(train_loader)):
            # adjust learning rate and momentum coefficient per iteration
            lr = adjust_learning_rate(optimizer, epoch + iteration / len(train_loader), config.TRAIN.EPOCHS, float(config.TRAIN.LR))
            # Concatenate qs and ks 
            q1, q2, k1, k2 = torch.cat(q1s, dim=0), torch.cat(q2s, dim=0), torch.cat(k1s, dim=0), torch.cat(k2s, dim=0)

            # compute loss
            loss = model.contrastive_loss(q1, k2) + model.contrastive_loss(q2, k1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # wandb_image = wandb.Image(
            #     torch.cat([x1[0], x2[0]], dim=1), 
            #     caption="Top: x1, Bottom: x2"
            # )

            # Report statistics
            # wandb.log({"augemented_images": wandb_image}})
            wandb.log({"loss": loss.item()})
            wandb.log({"learning_rate": lr})

            q1s, q2s, k1s, k2s = [], [], [], []
            del q1, q2, k1, k2, loss

        del x1, x2
        gc.collect()
        torch.cuda.empty_cache()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, max_epochs, base_lr):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < 10:
        lr = base_lr * epoch / 10 
    else:
        lr = base_lr * 0.5 * (1. + math.cos(math.pi * (epoch - 10) / (max_epochs - 10)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    args = parser.parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
