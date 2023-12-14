from tqdm import tqdm
import torch
import gc
import numpy as np

from training.inference import relation_infer
from utils.visualize_sample import create_sample_visual

def validate(train_loader, model, loss_function, epoch, device, config, wandb_run, similarity_metric):
    model.eval()
    num_iterations = len(train_loader)

    all_losses = {
        "nodes": 0,
        "edges": 0,
        "boxes": 0,
        "domain": 0,
        "total": 0,
    }

    # Reset similarity metric
    if similarity_metric is not None:
        similarity_metric.reset()

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
        all_losses["nodes"] += single_losses["nodes"]
        all_losses["edges"] += single_losses["edges"]
        all_losses["boxes"] += single_losses["boxes"]
        all_losses["domain"] += single_losses["domain"]
        all_losses["total"] += single_losses["total"]

        # Create graph
        pred_nodes, pred_edges = relation_infer(
            h.detach(), out, model, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN
        )

        # Add similarity information to metric
        if similarity_metric is not None:
            similarity_metric.update(srcs[-1], domains)

        # Log samples in first iteration
        if iteration == 0:
            sample_visuals = create_sample_visual(images, nodes, edges, pred_nodes, pred_edges)

        # Cleanup
        del images
        del nodes
        del edges
        del target
        del pred_nodes
        del pred_edges
        gc.collect()
        torch.cuda.empty_cache()

    # Compute average losses
    all_losses["nodes"] /= num_iterations
    all_losses["edges"] /= num_iterations
    all_losses["boxes"] /= num_iterations
    all_losses["domain"] /= num_iterations
    all_losses["total"] /= num_iterations

    return all_losses, sample_visuals