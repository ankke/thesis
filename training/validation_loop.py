from tqdm import tqdm
import torch
import gc
import numpy as np
from metrics.metric_map import BBoxEvaluator
from torch_geometric.data import Data

from training.inference import relation_infer
from utils.utils import ensure_format
from utils.visualize_sample import create_sample_visual
from greed.models import NormGEDModel



def validate(train_loader, model, loss_function, epoch, device, config, wandb_run, similarity_metric):
    ged_model = NormGEDModel(8, 1, 64, 64).to(device)
    ged_model.load_state_dict(torch.load('/home/anna_alex/greed/runlogs/OCTA/1707343788.9150653/best_model.pt'))

    model.eval()
    num_iterations = len(train_loader)

    all_results = {
        "node_loss": 0,
        "edge_loss": 0,
        "box_loss": 0,
        "domain_loss": 0,
        "total_loss": 0,
        "ged": 0,
        "ged_node": 0,
    }

    # Reset similarity metric
    if similarity_metric is not None:
        similarity_metric.reset()

    # Set Box evaluator for edge mAP
    metric_edge_map = BBoxEvaluator(['edge'], max_detections=200)

    # One epoch
    for iteration, (images, segs, nodes, edges, domains) in enumerate(tqdm(train_loader, leave=False, desc="Validation loop")):
        images = images.to(device,  non_blocking=False)
        nodes = [node.to(device,  non_blocking=False) for node in nodes]
        edges = [edge.to(device,  non_blocking=False) for edge in edges]
        domains = domains.to(device, non_blocking=False)
        target = {'nodes': nodes, 'edges': edges, 'domains': domains}

        # Compute parameters fdr domain adversarial
        p = float(iteration + epoch * config.TRAIN.EPOCHS) / config.TRAIN.EPOCHS / num_iterations
        alpha = (2. / (1. + np.exp(-10 * p)) - 1) * config.TRAIN.ALPHA_COEFF

        # Forward pass
        h, out, srcs, pred_backbone_domains, pred_instance_domains, interpolated_domains = model(images, alpha=alpha, domain_labels=domains)
        target["interpolated_domains"] = interpolated_domains

        # Compute losses
        with torch.no_grad():
            single_losses = loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)
        
        # Add losses to build average
        all_results["node_loss"] += single_losses["nodes"]
        all_results["edge_loss"] += single_losses["edges"]
        all_results["box_loss"] += single_losses["boxes"]
        all_results["domain_loss"] += single_losses["domain"]
        all_results["total_loss"] += single_losses["total"]

        # Create graph
        pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
            h.detach(), out, model, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN, map_=True
        )

        # Add similarity information to metric
        if similarity_metric is not None:
            similarity_metric.update(srcs[-1], domains)

        # Log samples in first iteration
        if iteration == 0:
            sample_visuals = create_sample_visual(images, nodes, edges, pred_nodes, pred_edges)

        # Compute edge map
        pred_edges_box = []
        for edges_, nodes_ in zip(pred_edges, pred_nodes):
            nodes_ = nodes_.cpu().numpy()
            edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
            pred_edges_box.append(edges_box)

        gt_edges_box = []
        for edges_, nodes_ in zip(edges, nodes):
            nodes_ , edges_ = nodes_.cpu().numpy(), edges_.cpu().numpy()
            edges_box = ensure_format(np.hstack([nodes_[edges_[:, 0]], nodes_[edges_[:, 1]]]))
            gt_edges_box.append(edges_box)

        metric_edge_map.add(
            pred_boxes=pred_edges_box,
            pred_classes=pred_edges_box_class,
            pred_scores=pred_edges_box_score,
            gt_boxes=gt_edges_box,
            gt_classes=[np.ones((edges_.shape[0],)) for edges_ in edges]
        )

        graphs = []
        for n, e in zip(nodes, edges):
            nodes_ones = torch.ones(n.size(0)).view(-1, 1)
            graph = Data(
                    x=nodes_ones,
                    edge_index=e.t(),
                    pos=n,
                )
            graphs.append(graph)
        
        pred_graphs = []
        for n, e in zip(pred_nodes, pred_edges):
            pred_nodes = n
            pred_edges = torch.squeeze(torch.tensor(e))
            pred_nodes_ones = torch.ones(n.size(0)).view(-1, 1)
            if  pred_edges.size(0) == 0:
                pred_edges = torch.empty((0, 2), dtype=torch.int64)

            pred_graph = Data(
                    x=pred_nodes_ones,
                    edge_index=pred_edges.t(),
                    pos=n,
                )
            pred_graphs.append(pred_graph)

        ged = ged_model.predict_inner(pred_graphs, graphs).tolist()
        all_results["ged"] += np.mean(ged)


        # Cleanup
        del images
        del nodes
        del edges
        del target
        del pred_nodes
        del pred_edges
        gc.collect()
        torch.cuda.empty_cache()

    # Compute mAP
    edge_metric_scores = metric_edge_map.eval()

    all_results["edge_mAP"] = edge_metric_scores['mAP_IoU_0.50_0.95_0.05_MaxDet_100'][0]
    all_results["edge_mAR"] = edge_metric_scores['mAR_IoU_0.50_0.95_0.05_MaxDet_100'][0]

    # Compute average losses
    all_results["node_loss"] /= num_iterations
    all_results["edge_loss"] /= num_iterations
    all_results["box_loss"] /= num_iterations
    all_results["domain_loss"] /= num_iterations
    all_results["total_loss"] /= num_iterations
    all_results["ged"] /= num_iterations

    return all_results, sample_visuals