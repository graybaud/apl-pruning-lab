"""Bridge between CastNet and apl-pruning.

Drop-in replacements for CastNet scorers that use APL formulas
instead of hardcoded PyTorch expressions.

Usage:
    from apl_pruning.castnet_bridge import accumulate_gradient_scores
    scores = accumulate_gradient_scores(model, ffn_layers, dataset, num_batches, device)
"""

import time
import torch
import numpy as np
from apl_pruning.scorers import score_layer


def _to_numpy(tensor):
    """Convert torch tensor to numpy float32."""
    return tensor.detach().float().cpu().numpy()


def _to_torch(array, device='cpu'):
    """Convert numpy array to torch tensor."""
    return torch.from_numpy(np.asarray(array)).float().to(device)


# ======================================================================
# Gradient: |W| x |grad|
# ======================================================================

def accumulate_gradient_scores(model, ffn_layers, dataset, num_batches, device):
    """Gradient scoring using APL formula: |W| x |grad|."""
    score_accum = {
        name: torch.zeros_like(module.weight.data, device='cpu')
        for name, module in ffn_layers
    }
    
    print(f"\n  [APL] Gradient scoring on {num_batches} batches...")
    t0, tokens_done = time.time(), 0
    
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        
        for name, module in ffn_layers:
            if module.weight.grad is not None:
                W_np = _to_numpy(module.weight.data)
                grad_np = _to_numpy(module.weight.grad)
                scores_np = score_layer("gradient", W=W_np, grad=grad_np)
                score_accum[name] += torch.from_numpy(scores_np).float()
        
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"    Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    
    print(f"  Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    return score_accum


# ======================================================================
# Wanda: |W| x mean(|act|)
# ======================================================================

def accumulate_wanda_scores(model, ffn_layers, dataset, num_batches, device):
    """Wanda scoring using APL formula: |W| x mean(|act|)."""
    act_accum = {
        name: torch.zeros(module.in_features, device='cpu')
        for name, module in ffn_layers
    }
    hooks, acts = [], {}
    
    def make_hook(n):
        def h(m, i, o):
            acts[n] = i[0].detach()
        return h
    
    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    print(f"\n  [APL] Wanda scoring on {num_batches} batches...")
    t0, tokens_done = time.time(), 0
    
    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches:
                break
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            
            for name in act_accum:
                if name in acts:
                    # Wanda accumulates L2 norm of activations per input dim
                    act_accum[name] += acts[name].float().pow(2).mean(dim=(0, 1)).sqrt().cpu()
            
            tokens_done += ids.numel()
            if (bidx + 1) % 10 == 0:
                print(f"    Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    
    for h in hooks:
        h.remove()
    
    print(f"  Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    
    # Apply APL formula: |W| x mean(|act|)
    scores = {}
    for name, module in ffn_layers:
        W_np = _to_numpy(module.weight.data)
        act_np = _to_numpy(act_accum[name] / num_batches)
        act_np = act_np.reshape(1, -1)  # (in_features,) -> (1, in_features)
        scores_np = score_layer("wanda", W=W_np, act=act_np)
        scores[name] = torch.from_numpy(scores_np).float()
    
    return scores


# ======================================================================
# Magnitude: |W|
# ======================================================================

def accumulate_magnitude_scores(model, ffn_layers, dataset, num_batches, device):
    """Magnitude scoring using APL formula: |W|."""
    print(f"\n  [APL] Magnitude scoring ({len(ffn_layers)} layers)...")
    scores = {}
    for name, module in ffn_layers:
        W_np = _to_numpy(module.weight.data)
        scores_np = score_layer("magnitude", W=W_np)
        scores[name] = torch.from_numpy(scores_np).float()
    print(f"  Done. {len(scores)} layers scored.")
    return scores


# ======================================================================
# GPS Local: direction x selectivity x distortion
# ======================================================================

def accumulate_gps_scores(model, ffn_layers, dataset, num_batches, device,
                           sample_tokens=300, max_distortion_tokens=100):
    """GPS scoring using APL formula with torch data collection."""
    print(f"\n  [APL] GPS Local scoring...")
    
    layer_inputs = {name: [] for name, _ in ffn_layers}
    layer_outputs = {name: [] for name, _ in ffn_layers}
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[f"{name}_in"] = input[0].detach()
            activations[f"{name}_out"] = output.detach()
        return hook_fn
    
    for name, module in ffn_layers:
        hooks.append(module.register_forward_hook(make_hook(name)))
    
    t0 = time.time()
    tokens_collected = 0
    
    with torch.no_grad():
        for bidx, batch in enumerate(dataset):
            if bidx >= num_batches or tokens_collected >= sample_tokens:
                break
            ids = batch['input_ids']
            if isinstance(ids, list):
                ids = torch.tensor(ids)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            _ = model(ids)
            
            for name, _ in ffn_layers:
                key_in, key_out = f"{name}_in", f"{name}_out"
                if key_in in activations and key_out in activations:
                    x_in = activations[key_in]
                    x_out = activations[key_out]
                    if x_in.dim() == 3:
                        x_in = x_in.reshape(-1, x_in.shape[-1])
                        x_out = x_out.reshape(-1, x_out.shape[-1])
                    layer_inputs[name].append(x_in.cpu())
                    layer_outputs[name].append(x_out.cpu())
            
            tokens_collected += ids.numel()
    
    for h in hooks:
        h.remove()
    
    print(f"  Collected {tokens_collected} tokens in {time.time()-t0:.0f}s")
    
    gps_scores = {}
    active_layers = [n for n, _ in ffn_layers if n in layer_inputs and len(layer_inputs[n]) > 0]
    weight_dict = {name: module.weight.data.float() for name, module in ffn_layers}
    
    for layer_idx, name in enumerate(active_layers, 1):
        x_in = torch.cat(layer_inputs[name], dim=0)[:sample_tokens]
        x_out = torch.cat(layer_outputs[name], dim=0)[:sample_tokens]
        if x_in.shape[0] < 10:
            continue
        
        W = weight_dict[name]
        out_dim = W.shape[0]
        
        # Prepare variables for APL formula
        W_np = _to_numpy(W)
        act_np = _to_numpy(x_in.float() @ W.T)  # (N, out_dim) - neuron activations
        act_in_np = _to_numpy(x_in.float())
        act_out_np = _to_numpy(x_out.float())
        
        # Use per-neuron GPS formula
        scores_np = score_layer("gps_local",
            W=W_np,
            act=act_np,
            act_in=act_in_np,
            act_out=act_out_np
        )
        
        # Handle shape: GPS returns per-neuron scores, expand to full matrix
        if scores_np.ndim == 1:
            scores_np = scores_np.reshape(-1, 1).repeat(W.shape[1], axis=1).T
        
        gps = torch.from_numpy(scores_np).float()
        gps = torch.clamp(gps, min=0.0)
        gps_max = gps.max()
        if gps_max > 0:
            gps = gps / gps_max
        
        print(f"  [{layer_idx}/{len(active_layers)}] {name}: [{out_dim}x{W.shape[1]}] "
              f"GPS=[{gps.min():.4f},{gps.max():.4f}]")
        
        gps_scores[name] = gps.cpu()
    
    print(f"  GPS Local complete. {len(gps_scores)} layers scored.")
    return gps_scores


# ======================================================================
# Chain: gradient with downstream importance
# ======================================================================

def accumulate_chain_scores(model, ffn_pairs, dataset, num_batches, device):
    """Chain scoring: |W| x |grad| with downstream importance bonus."""
    s1, s2 = {}, {}
    
    print(f"\n  [APL] Chain scoring on {num_batches} batches...")
    t0, tokens_done = time.time(), 0
    
    for bidx, batch in enumerate(dataset):
        if bidx >= num_batches:
            break
        ids = batch['input_ids']
        if isinstance(ids, list):
            ids = torch.tensor(ids)
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        
        model.zero_grad()
        model(ids, labels=ids).loss.backward()
        
        done = set()
        for n1, fc1, n2, fc2 in ffn_pairs:
            # fc2 scoring: |W2| x |grad2|
            if fc2.weight.grad is not None and n2 not in done:
                W2_np = _to_numpy(fc2.weight.data)
                grad2_np = _to_numpy(fc2.weight.grad)
                s2[n2] = s2.get(n2, 0) + score_layer("gradient", W=W2_np, grad=grad2_np)
                done.add(n2)
            
            # fc1 scoring: |W1| x |grad1| x (1 + importance_from_fc2)
            if fc1.weight.grad is not None and fc2.weight.grad is not None:
                W1_np = _to_numpy(fc1.weight.data)
                grad1_np = _to_numpy(fc1.weight.grad)
                base = score_layer("gradient", W=W1_np, grad=grad1_np)
                
                # Downstream importance: aggregated |grad2| per output neuron
                imp = _to_numpy(fc2.weight.grad.abs().sum(dim=0))
                imp = imp / (imp.max() + 1e-8)
                # Broadcast: base is (out_fc1, in_fc1), imp is (out_fc1,) -> (out_fc1, in_fc1)
                bonus = base * (1.0 + imp.reshape(-1, 1))
                s1[n1] = s1.get(n1, 0) + bonus
        
        tokens_done += ids.numel()
        if (bidx + 1) % 10 == 0:
            print(f"    Batch {bidx+1}/{num_batches}  {tokens_done} tokens")
    
    print(f"  Done. {tokens_done} tokens in {time.time()-t0:.0f}s")
    
    # Convert to torch
    result = {}
    for d in [s2, s1]:
        for k, v in d.items():
            result[k] = torch.from_numpy(np.asarray(v)).float()
    
    return result


# ======================================================================
# Compare: run multiple methods and compare scores
# ======================================================================

def compare_all_methods(W, act=None, grad=None):
    """Compare all APL methods on a single layer. Returns dict of scores."""
    from apl_pruning.scorers import compare_methods
    
    methods = ["magnitude", "wanda", "gradient", "direction", 
               "direction_per_neuron", "softmax_grad", "norm_ratio", "threshold"]
    variables = {"W": _to_numpy(W) if isinstance(W, torch.Tensor) else W}
    if act is not None:
        variables["act"] = _to_numpy(act) if isinstance(act, torch.Tensor) else act
    if grad is not None:
        variables["grad"] = _to_numpy(grad) if isinstance(grad, torch.Tensor) else grad
    
    # Filter methods that have all required variables
    from apl_pruning.scorers import METHODS
    available = [m for m in methods if m in METHODS and 
                 all(v in variables for v in METHODS[m]["variables"])]
    
    return compare_methods(available, **variables)
