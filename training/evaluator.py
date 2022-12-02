import os
import gc

import torch
from monai.engines import SupervisedEvaluator
from monai.handlers import StatsHandler, CheckpointSaver, TensorBoardStatsHandler, TensorBoardImageHandler
from metrics.loss_metric import MeanLoss
from metrics.metric_smd import MeanSMD
from training.inference import relation_infer

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Sequence, Union
from monai.config import IgniteInfo
from monai.engines.utils import default_metric_cmp_fn, default_prepare_batch
from monai.inferers import Inferer, SimpleInferer
from monai.transforms import Transform
from monai.utils import ForwardMode, min_version, optional_import
if TYPE_CHECKING:
    from ignite.engine import EventEnum
    from ignite.metrics import Metric
else:
    Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
    Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
    EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")

from data.visualize_sample import create_sample_visual

# Define customized evaluator
class RelationformerEvaluator(SupervisedEvaluator):
    def __init__(
        self,
        device: torch.device,
        val_data_loader: Union[Iterable, DataLoader],
        network: torch.nn.Module,
        loss_function=None,
        epoch_length: Optional[int] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = default_prepare_batch,
        iteration_update: Optional[Callable] = None,
        inferer: Optional[Inferer] = None,
        postprocessing: Optional[Transform] = None,
        key_val_metric: Optional[Dict[str, Metric]] = None,
        additional_metrics: Optional[Dict[str, Metric]] = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        val_handlers: Optional[Sequence] = None,
        amp: bool = False,
        mode: Union[ForwardMode, str] = ForwardMode.EVAL,
        event_names: Optional[List[Union[str, EventEnum]]] = None,
        event_to_attr: Optional[dict] = None,
        decollate: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            device=device,
            val_data_loader=val_data_loader,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            prepare_batch=prepare_batch,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_val_metric=key_val_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            val_handlers=val_handlers,
            amp=amp,
            mode=mode,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            network=network,
            inferer=SimpleInferer() if inferer is None else inferer,
        )

        self.loss_function = loss_function
        self.config = kwargs.pop('config')
        
    def _iteration(self, engine, batchdata):
        images, nodes, edges = batchdata[0], batchdata[2], batchdata[3]
        
        # # inputs, targets = self.get_batch(batchdata, image_keys=IMAGE_KEYS, label_keys="label")
        # # inputs = torch.cat(inputs, 1)
        images = images.to(engine.state.device,  non_blocking=False)
        nodes = [node.to(engine.state.device,  non_blocking=False) for node in nodes]
        edges = [edge.to(engine.state.device,  non_blocking=False) for edge in edges]
        target = {'nodes': nodes, 'edges': edges}

        self.network.eval()
        
        h, out, srcs = self.network(images, seg=False)

        losses = self.loss_function(
            h.clone().detach(),
            {'pred_logits': out['pred_logits'].clone().detach(), 'pred_nodes': out["pred_nodes"].clone().detach()},
            {'nodes': [node.clone().detach() for node in nodes], 'edges': [edge.clone().detach() for edge in edges]}
        )

        pred_nodes, pred_edges = relation_infer(
            h.detach(), out, self.network, self.config.MODEL.DECODER.OBJ_TOKEN, self.config.MODEL.DECODER.RLN_TOKEN
        )
        
        # if self.config.TRAIN.SAVE_VAL:
        #     root_path = os.path.join(self.config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (self.config.log.exp_name, self.config.DATA.SEED), 'val_samples')
        #     if not os.path.exists(root_path):
        #         os.makedirs(root_path)
        #     for i, (node, edge, pred_node, pred_edge) in enumerate(zip(nodes, edges, pred_nodes, pred_edges)):
        #         path = os.path.join(root_path, "ref_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
        #         save_input(path, i, images[i,0,...].cpu().numpy(), node.cpu().numpy(), edge.cpu().numpy())
        #         path = os.path.join(root_path, "pred_epoch_"+str(engine.state.epoch).zfill(3)+"_iteration_"+str(engine.state.iteration).zfill(5))
        #         save_output(path, i, pred_node.cpu().numpy(), pred_edge.cpu().numpy())

        gc.collect()
        torch.cuda.empty_cache()
        return {"images": images, "nodes": nodes, "edges": edges, "pred_nodes": pred_nodes, "pred_edges": pred_edges, "loss": losses}


def build_evaluator(val_loader, net, loss, optimizer, scheduler, writer, config, device, early_stop_handler):
    """[summary]

    Args:
        val_loader ([type]): [description]
        net ([type]): [description]
        device ([type]): [description]

    Returns:
        [type]: [description]
    """
    val_handlers = [
        early_stop_handler,
        StatsHandler(output_transform=lambda x: None),
        CheckpointSaver(
            save_dir=os.path.join(config.TRAIN.SAVE_PATH, "runs", '%s_%d' % (config.log.exp_name, config.DATA.SEED),
                                  './models'),
            save_dict={"net": net, "optimizer": optimizer, "scheduler": scheduler},
            save_key_metric=True,
            key_metric_n_saved=1,
            save_interval=1,
            key_metric_negative_sign=True
        ),
        TensorBoardStatsHandler(
            writer,
            tag_name="val_total_loss",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: scheduler.last_epoch
        ),
        TensorBoardImageHandler(
            writer,
            epoch_level=True,
            interval=1,
            # max_channels=3,>
            output_transform=lambda x: create_sample_visual(x)
        )
    ]

    # val_post_transform = Compose(
    #     [AsDiscreted(keys=("pred", "label"),
    #     argmax=(True, False),
    #     to_onehot=True,
    #     n_classes=N_CLASS)]
    # )

    evaluator = RelationformerEvaluator(
        config=config,
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SimpleInferer(),
        # post_transform=val_post_transform,
        key_val_metric={
            "val_total_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["total"],
            ),
        },
        additional_metrics={
            "val_smd": MeanSMD(
                output_transform=lambda x: (x["nodes"], x["edges"], x["pred_nodes"], x["pred_edges"]),
            ),
            "val_node_loss": MeanLoss(
                output_transform = lambda x: x["loss"]["nodes"],
            ),
            "val_class_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["class"],
            ),
            "val_edge_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["edges"],
            ),
            "val_card_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["cards"],
            ),
        },
        val_handlers=val_handlers,
        amp=False,
        loss_function=loss
    )

    """"val_node_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["nodes"],
            ),
            "val_class_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["class"],
            ),
            "val_edge_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["edges"],
            ),
            "val_card_loss": MeanLoss(
                output_transform=lambda x: x["loss"]["cards"],
            ),
            "val_smd": MeanSMD(
                output_transform=lambda x: x,
            ),"""

    return evaluator


def write_metric(engine, writer, metric_name):
    writer.add_scalar(f"val_{metric_name}_loss", engine.state.output["loss"][metric_name].item(), engine.state.epoch)
    writer.flush()

