import os
from monai.engines import SupervisedTrainer
from monai.inferers import SimpleInferer
from monai.handlers import LrScheduleHandler, ValidationHandler, StatsHandler, TensorBoardStatsHandler, \
    CheckpointSaver
import torch
import gc
import numpy as np

from metrics.similarity import batch_cka, batch_cosine, batch_euclidean, upsample_examples
from metrics.svcca import get_cca_similarity, robust_cca_similarity


# define customized trainer
class RelationformerTrainer(SupervisedTrainer):

    def _iteration(self, engine, batchdata):
        images, seg, nodes, edges, domains = batchdata[0], batchdata[1], batchdata[2], batchdata[3], batchdata[4]
        # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # inputs = torch.cat(inputs, 1)

        images = images.to(engine.state.device,  non_blocking=False)
        seg = seg.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        domains = domains.to(engine.state.device, non_blocking=False)
        target = {'nodes': nodes, 'edges': edges, 'domains': domains}

        self.network[0].train()
        self.network[1].eval()
        self.optimizer.zero_grad()

        epoch = engine.state.epoch
        iteration = engine.state.iteration
        p = float(iteration + epoch * engine.state.epoch_length) / engine.state.max_epochs / engine.state.epoch_length
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        h, out, srcs, pred_domains = self.network[0](images, seg=False, alpha=alpha)

        # _, _, seg_srcs = self.network[1](seg, seg=True)
        
        # # valid_token = torch.argmax(out['pred_logits'], -1)
        # # valid_token = torch.sigmoid(nodes_prob[...,3])>0.5
        # # print('valid_token number', valid_token.sum(1))

        # # pred_nodes, pred_edges = relation_infer(h, out, self.network.relation_embed)

        #print(torch.concat([torch.exp(pred_domains), domains.unsqueeze(1)], axis=1))

        losses = self.loss_function(h, out, target, pred_domains)
        
        # loss_feat = 0
        # for i in range(len(seg_srcs)):
        #     loss_feat += F.l1_loss(srcs[i], seg_srcs[i].detach())

        # losses.update({'feature':loss_feat})
        # losses['total'] = losses['total']+loss_feat
        # Clip the gradient
        # clip_grad_norm_(
        #     self.network.parameters(),
        #     max_norm=GRADIENT_CLIP_L2_NORM,
        #     norm_type=2,
        # )
        losses['total'].backward()

        # if 0.1 > 0:
        #     _ = torch.nn.utils.clip_grad_norm_(self.network[0].parameters(), 0.1)
        # else:
        #     _ = get_total_grad_norm(self.network[0].parameters(), 0.1)
    
        self.optimizer.step()
        
        gc.collect()
        torch.cuda.empty_cache()

        # Filtering for each domain and flatten the spatial dimensions of X (batch, channel, height, widths) and Y while keeping the channels
        X = srcs[-1][domains == 0].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu().numpy()
        Y = srcs[-1][domains == 1].clone().permute(0,2,3,1).flatten(start_dim=0, end_dim=2).detach().cpu().numpy()

        X_up, Y_up = upsample_examples(srcs[-1][domains == 0].detach().cpu().numpy(),srcs[-1][domains == 1].detach().cpu().numpy())
        # Permute and flatten the umsapled numpy arrays
        X_up = np.moveaxis(X_up, 1,3).reshape(-1, X_up.shape[1])
        Y_up = np.moveaxis(Y_up, 1,3).reshape(-1, Y_up.shape[1])

        cka_similarity = batch_cka(X_up,Y_up)
        cosine_similarity = batch_cosine(X,Y)
        euclidean_distance = batch_euclidean(X,Y)

        #svcca_similarity = robust_cca_similarity(X_up,Y_up, threshold=0.98, compute_dirns=False, verbose=False, epsilon=1e-8)["mean"]
        svcca_similarity = 0
        similarities = {
            "cka": cka_similarity,
            "cosine": cosine_similarity,
            "euclidean": euclidean_distance,
            "svcca": svcca_similarity
        }

        return {"images": images, "points": nodes, "edges": edges, "loss": losses, "similarities": similarities}


def build_trainer(train_loader, net, seg_net, loss, optimizer, scheduler, writer,
                  evaluator, config, device, fp16=False):
    """[summary]

    Args:
        train_loader ([type]): [description]
        net ([type]): [description]
        loss ([type]): [description]
        optimizer ([type]): [description]
        evaluator ([type]): [description]
        scheduler ([type]): [description]
        max_epochs ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    train_handlers = [
        LrScheduleHandler(
            lr_scheduler=scheduler,
            print_lr=True,
            epoch_level=True,
        ),
        ValidationHandler(
            validator=evaluator,
            interval=config.TRAIN.VAL_INTERVAL,
            epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x["loss"]["total"]
        ),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                  './models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_interval=5,
            n_saved=5
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="cka_similarity",
            output_transform=lambda x: x["similarities"]["cka"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="svcca_similarity",
            output_transform=lambda x: x["similarities"]["svcca"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="euclidean_distance",
            output_transform=lambda x: x["similarities"]["euclidean"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="cosine_similarity",
            output_transform=lambda x: x["similarities"]["cosine"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="classification_loss",
            output_transform=lambda x: x["loss"]["class"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="node_loss",
            output_transform=lambda x: x["loss"]["nodes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="edge_loss",
            output_transform=lambda x: x["loss"]["edges"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="box_loss",
            output_transform=lambda x: x["loss"]["boxes"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="card_loss",
            output_transform=lambda x: x["loss"]["cards"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="domain_loss",
            output_transform=lambda x: x["loss"]["domain"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="total_loss",
            output_transform=lambda x: x["loss"]["total"],
            global_epoch_transform=lambda x: scheduler.last_epoch,
            epoch_log=True,
            epoch_interval=1
        )
    ]
    # train_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    loss.to(device)

    trainer = RelationformerTrainer(
        device=device,
        max_epochs=config.TRAIN.EPOCHS,
        train_data_loader=train_loader,
        network=[net, seg_net],
        optimizer=optimizer,
        loss_function=loss,
        inferer=SimpleInferer(),
        # post_transform=train_post_transform,
        # key_train_metric={
        #     "train_mean_dice": MeanDice(
        #         include_background=False,
        #         output_transform=lambda x: (x["pred"], x["label"]),
        #     )
        # },
        train_handlers=train_handlers,
        # amp=fp16,
    )

    return trainer

