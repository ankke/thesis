from tqdm import tqdm
import torch
import gc
import numpy as np

def train(train_loader, model, optimizer, loss_function, epoch, device, config, wandb_run, similarity_metric):
    model.train()
    num_iterations = len(train_loader)

    # Reset similarity metric
    if similarity_metric is not None:
        similarity_metric.reset()

    # One epoch
    for iteration, (images, segs, nodes, edges, domains) in enumerate(tqdm(train_loader, leave=False, desc="Training loop")):
        optimizer.zero_grad()

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
        losses = loss_function(h, out, target, pred_backbone_domains, pred_instance_domains)

        # Backward pass
        losses['total'].backward()
        optimizer.step()

        # Add similarity information to metric
        if similarity_metric is not None:
            similarity_metric.update(srcs[-1], domains)

        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({"train": {
                "node_loss": losses["nodes"],
                "edge_loss": losses["edges"],
                "box_loss": losses["boxes"],
                "domain_loss": losses["domain"],
                "total_loss": losses["total"],
                "alpha": alpha,
                "step": num_iterations*epoch + iteration,
            }})

        # Cleanup
        del images
        del nodes
        del edges
        del target
        gc.collect()
        torch.cuda.empty_cache()





