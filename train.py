from functools import partial
import os
import yaml
import json
import shutil
from argparse import ArgumentParser
import torch
import torch.nn as nn
from monai.data import DataLoader
from data.dataset_mixed import build_mixed_data
from data.dataset_road_network import build_road_network_data
from data.dataset_synthetic_eye_vessels import build_synthetic_vessel_network_data
from data.dataset_real_eye_vessels import build_real_vessel_network_data
from metrics.similarity import SimilarityMetricPCA, batch_cka
from metrics.svcca import robust_cca_similarity
from training.train_loop import train
from models import build_model
from training.validation_loop import validate
from utils.utils import image_graph_collate_road_network
from models.matcher import build_matcher
from training.losses import EDGE_SAMPLING_MODE, SetCriterion
from tqdm import tqdm
import wandb
from sklearn.model_selection import KFold
from greed.models import NormGEDModel


parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yml) containing the hyper-parameters for training. '
                         'If None, use the nnU-Net config. See /config for examples.')
parser.add_argument('--resume', default=None,
                    help='checkpoint of the last epoch of the model')
parser.add_argument('--restore_state', dest='restore_state', help='whether the state should be restored', action='store_true')
parser.add_argument('--device', default='cuda',
                    help='device to use for training')
parser.add_argument('--recover_optim', default=False, action="store_true",
                    help="Whether to restore optimizer's state. Only necessary when resuming training.")
parser.add_argument('--exp_name', dest='exp_name', help='name of the experiment', type=str,required=True)
parser.add_argument('--pretrain_seg', default=False, action="store_true",
                    help="Whether to pretrain on segs instead of raw images")
parser.add_argument('--no_strict_loading', default=False, action="store_true",
                    help="Whether the model was pretrained with domain adversarial. If true, the checkpoint will be loaded with strict=false")
parser.add_argument('--sspt', default=False, action="store_true",
                    help="Whether the model was pretrained with self supervised pretraining. If true, the checkpoint will be loaded accordingly. Only combine with resume.")
parser.add_argument('--disable_wandb', default=False, action='store_true',
                    help='disable wandb logging')
parser.add_argument('--folds', default=1, help='Number of folds for cross-validation')


class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)


def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

os.environ['WANDB_API_KEY'] = 'a970cf6fefec316437e0f257a5535fc745f1691d'


def main(args):
    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(args.exp_name)
    config = dict2obj(config)
    config.log.exp_name = args.exp_name

    # Create directory for this run
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

    # Enable torch optimization
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device == 'cuda' else torch.device("cpu")

    # Determine dataset type
    if config.DATA.DATASET == 'road_dataset':
        build_dataset_function = partial(build_road_network_data, max_samples=config.DATA.NUM_SOURCE_SAMPLES)
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'synthetic_eye_vessel_dataset':
        build_dataset_function = partial(build_synthetic_vessel_network_data, max_samples=config.DATA.NUM_SOURCE_SAMPLES)
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'real_eye_vessel_dataset':
        build_dataset_function = partial(build_real_vessel_network_data, max_samples=config.DATA.NUM_SOURCE_SAMPLES)
        config.DATA.MIXED = False
    elif config.DATA.DATASET == 'mixed_road_dataset' or config.DATA.DATASET == 'mixed_synthetic_eye_vessel_dataset' or config.DATA.DATASET == "mixed_real_eye_vessel_dataset":
        build_dataset_function = partial(build_mixed_data, mode="train")
        config.DATA.MIXED = True
    else:
        raise ValueError("Invalid dataset")

    # Create dataset
    train_ds = build_dataset_function(
        config, mode='train', use_grayscale=args.pretrain_seg
    )

    # Initialize the k-fold cross validation
    folds = int(args.folds)
    if folds > 1:
        kf = KFold(n_splits=folds, shuffle=True, random_state=config.DATA.SEED)
        iterator = kf.split(train_ds)
        max_edge_maps = torch.zeros(size=[folds], dtype=torch.float)
    else:
        ds_range = torch.randperm(len(train_ds))
        # Randomly select 80% of the ids in ds_range for training and the rest for validation
        train_idx = ds_range[:int(0.8*len(train_ds))]
        val_idx = ds_range[int(0.8*len(train_ds)):]

        iterator = [(train_idx, val_idx)]

    for fold, (train_idx, val_idx) in enumerate(iterator):
        # Create dataloader
        train_loader = DataLoader(train_ds,
                                batch_size=config.DATA.BATCH_SIZE,
                                num_workers=config.DATA.NUM_WORKERS,
                                shuffle=False,
                                collate_fn=image_graph_collate_road_network,
                                pin_memory=True,
                                sampler=torch.utils.data.SubsetRandomSampler(train_idx)
        )

        val_loader = DataLoader(train_ds,
                                batch_size=config.DATA.BATCH_SIZE,
                                shuffle=False,
                                num_workers=config.DATA.NUM_WORKERS,
                                collate_fn=image_graph_collate_road_network,
                                pin_memory=True,
                                sampler=torch.utils.data.SubsetRandomSampler(val_idx)
        )

        print("Training on %d samples, validating on %d samples" % (len(train_idx), len(val_idx)))

        # Set edge sampling mode
        if config.TRAIN.EDGE_SAMPLING_MODE == "none":
            edge_sampling_mode = EDGE_SAMPLING_MODE.NONE
        elif config.TRAIN.EDGE_SAMPLING_MODE == "up":
            edge_sampling_mode = EDGE_SAMPLING_MODE.UP
        elif config.TRAIN.EDGE_SAMPLING_MODE == "down":
            edge_sampling_mode = EDGE_SAMPLING_MODE.DOWN
        else:
            raise ValueError("Invalid edge sampling mode")
        
        ged_model = NormGEDModel(8, 1, 64, 64).to(device)
        ged_model.load_state_dict(torch.load('/home/anna_alex/greed/runlogs/OCTA/1707343788.9150653/best_model.pt'))

        # Create model, and loss, and utilities
        model = build_model(config).to(device)
        matcher = build_matcher(config)
        loss = SetCriterion(
            config,
            matcher,
            model,
            ged_model,
            num_edge_samples=config.TRAIN.NUM_EDGE_SAMPLES,
            edge_sampling_mode=edge_sampling_mode,
            domain_class_weight=torch.tensor(config.TRAIN.DOMAIN_WEIGHTING, device=device)
        )

        # Create validation loss criterion
        val_loss = SetCriterion(config, matcher, model, ged_model, num_edge_samples=9999, edge_sampling_mode=EDGE_SAMPLING_MODE.NONE)

        # Set learning rates according to config for different parts of the network
        param_dicts = [
            {
                "params":
                    [p for n, p in model.named_parameters()
                    if not match_name_keywords(n, ["encoder.0"]) and not match_name_keywords(n, ['reference_points', 'sampling_offsets']) and not match_name_keywords(n, ["domain_discriminator"]) and p.requires_grad],
                "lr": float(config.TRAIN.LR),
                "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
            },
            {
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ["encoder.0"]) and p.requires_grad],
                "lr": float(config.TRAIN.LR_BACKBONE),
                "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
            },
            {
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ['reference_points', 'sampling_offsets']) and p.requires_grad],
                "lr": float(config.TRAIN.LR)*0.1,
                "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
            },
            {
                "params": [p for n, p in model.named_parameters() if match_name_keywords(n, ['domain_discriminator']) and p.requires_grad],
                "lr": float(config.TRAIN.LR_DOMAIN),
                "weight_decay": float(config.TRAIN.WEIGHT_DECAY)
            }
        ]

        # Create optmizer
        optimizer = torch.optim.AdamW(
            param_dicts, lr=float(config.TRAIN.LR), weight_decay=float(config.TRAIN.WEIGHT_DECAY)
        )

        # Print learning rates for debugging 
        for param_group in optimizer.param_groups:
            print(f'lr: {param_group["lr"]}, number of params: {len(param_group["params"])}')

        # Create Learning rate schedule
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_DROP)

        # Load existing model
        last_epoch = 0
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')

            if args.sspt:
                checkpoint['state_dict'] = {k[17:]: v for k, v in checkpoint['state_dict'].items() if k.startswith("momentum_encoder")}
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            else:
                model.load_state_dict(checkpoint['net'], strict=not args.no_strict_loading)
                if args.recover_optim:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if args.restore_state:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    last_epoch = scheduler.last_epoch
                    scheduler.step_size = config.TRAIN.LR_DROP


        # Create wandb run
        if not args.disable_wandb:
            # start a new wandb run to track this script
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project="relationformer",
                
                # track hyperparameters and run metadata
                config={
                    "learning_rate": float(config.TRAIN.LR),
                    "epochs": config.TRAIN.EPOCHS,
                    "batch_size": config.DATA.BATCH_SIZE,
                    "dataset": config.DATA.DATASET,
                    "exp_name": args.exp_name,
                    "fold": fold,
                }
            )
        else:
            wandb_run = None

        # Setup similarity metrics
        if config.DATA.MIXED:
            cca_similarity_metric = SimilarityMetricPCA( 
                similarity_function=lambda X, Y: robust_cca_similarity(X,Y, threshold=0.98, compute_dirns=False, verbose=False, epsilon=1e-8)["mean"][0],
                base_metric=None
            )
            cka_similarity_metric = SimilarityMetricPCA(
                similarity_function=batch_cka,
                base_metric=cca_similarity_metric
            )
        else:
            cca_similarity_metric = None
            cka_similarity_metric = None

        for epoch in tqdm(range(last_epoch, config.TRAIN.EPOCHS), desc="Epochs"):
            # Run epoch
            train(train_loader, model, optimizer, loss, epoch, device, config, wandb_run, cca_similarity_metric)

            logging = {
                "train": {
                    "base_lr": scheduler.get_last_lr()[0],
                },
                "validation": {},
                "epoch": epoch
            }

            if epoch % config.TRAIN.VAL_INTERVAL == 0:

                # Compute Training Similarity Metrics
                if config.DATA.MIXED:
                    logging["train"]["train_cca_similarity"] = cca_similarity_metric.compute()
                    logging["train"]["train_cka_similarity"] = cka_similarity_metric.compute()

                # Run validation
                validation_results, sample_visuals = validate(val_loader, model, val_loss, epoch, device, config, wandb_run, cca_similarity_metric)

                # Update best edge mAP
                if folds > 1 and validation_results["edge_mAP"] > max_edge_maps[fold]:
                    max_edge_maps[fold] = validation_results["edge_mAP"]

                # Compute Training Similarity Metrics
                if config.DATA.MIXED:
                    logging["validation"]["val_cca_similarity"] = cca_similarity_metric.compute()
                    logging["validation"]["val_cka_similarity"] = cka_similarity_metric.compute()

                logging["validation"].update(validation_results)
                logging["validation"]["sample_visuals"] = wandb.Image(sample_visuals)

                # Log whole-epoch validation stuff
                if wandb_run is not None:
                    wandb.log(logging)

                # Save checkpoint
                checkpoint = {
                    'net': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(checkpoint, os.path.join(exp_path, "checkpoint_%d.pth" % epoch))

            # Do scheduler step
            scheduler.step()

        if not args.disable_wandb:
            wandb.finish()

    # Print best and average edge mAP and std
    if folds > 1:
        print("Best edge mAP: %f" % max_edge_maps.max())
        print("Average edge mAP: %f" % max_edge_maps.mean())
        print("Edge mAP std: %f" % max_edge_maps.std())


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

if __name__ == '__main__':
    args = parser.parse_args()

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    main(args)
