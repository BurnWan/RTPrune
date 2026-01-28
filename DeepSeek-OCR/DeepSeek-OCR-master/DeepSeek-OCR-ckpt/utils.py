import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap  
import einops as ein
import math

def draw_line_chart(output_path, importance, h, w):
    base_size = h * w
    sim_map = importance.reshape(base_size, base_size).unsqueeze(0).unsqueeze(0)
    sim_map = F.interpolate(sim_map, size=(base_size, base_size), mode='nearest') 
    
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    
    sim_map_2d = sim_map.cpu().squeeze().detach().numpy()  

    plt.figure(figsize=(8, 8))  
    colors = [(1, 1, 1), (0, 0, 1)]
    white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", colors, N=256)
    
    im = plt.imshow(
        sim_map_2d, 
        cmap=white_to_blue,  
        vmin=0, vmax=1,   
        extent=[0, base_size-1, base_size-1, 0]  
    )
    
    
    # plt.xlabel('X Index')
    # plt.ylabel('Y Index')
    plt.xticks(np.arange(0, base_size, 50))  
    plt.yticks(np.arange(0, base_size, 50))

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  

def draw_imp_img(image_ori, output_path, importance, base_size, h, w):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    sim_map = importance.reshape(h, w).unsqueeze(0).unsqueeze(0)

    sim_map = F.interpolate(sim_map, size=(base_size, base_size), mode='nearest')
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min())
    # to cpu
    sim_map = sim_map.cpu()
    sim_map = np.uint8(sim_map.squeeze(0).squeeze(0).detach() * 255)

    jet_colormap = plt.get_cmap('jet')
    sim_map_colored = jet_colormap(sim_map)
    sim_map_colored = np.uint8(sim_map_colored * 255)

    jet_colormap = plt.get_cmap('jet')
    sim_map_colored = jet_colormap(sim_map)
    sim_map_colored = np.uint8(sim_map_colored * 255)

    alpha = 0.5
    img_tensor = image_ori.squeeze(0)
    img_np = img_tensor.permute(1, 2, 0).cpu().to(torch.float).detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)

    overlay = np.uint8(img_np * (1 - alpha) + sim_map_colored[:, :, :3] * alpha)
    plt.imsave(output_path, overlay)

def draw_rele_new(image_ori, output_path, relevance, base_size, h, w):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)


    sim_map = relevance.reshape(h, w).unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
    sim_map = F.interpolate(sim_map, size=(base_size, base_size), mode='nearest')  
    sim_map = sim_map.cpu().detach().squeeze(0).squeeze(0)  


    alpha_map = np.where(sim_map == 1, 0.0, 0.5)  
    if torch.is_tensor(alpha_map):
        alpha_map = alpha_map.numpy()

    gray_value = (158, 170, 209)  
    gray_img = np.ones((base_size, base_size, 3), dtype=np.uint8) * gray_value 

    img_tensor = image_ori.squeeze(0)
    img_np = img_tensor.permute(1, 2, 0).cpu().to(torch.float).detach().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_np = (img_np * 255).astype(np.uint8)
    assert img_np.shape[:2] == (base_size, base_size), f"原图尺寸{img_np.shape[:2]}与目标尺寸{base_size}不匹配"

    alpha_map_3d = np.expand_dims(alpha_map, axis=-1).repeat(3, axis=-1)

    overlay = np.uint8(img_np * (1 - alpha_map_3d) + gray_img * alpha_map_3d)

    plt.imsave(output_path, overlay)
                
def dynamic_ratio_by_similarity(prunable_features):
    x = prunable_features / prunable_features.norm(dim=1, keepdim=True)
    N = x.shape[0]

    sim = x @ x.T
    sim.fill_diagonal_(0.0)
    overlap = (sim ** 2).sum(dim=1) / (N - 1)
    overlap = torch.mean(overlap)

    dynamic_ratio = 1 - overlap

    return  dynamic_ratio

def dynamic_ratio_by_info(image_ori, h, w, myway="sobel"):
    B, C, H, W = image_ori.shape
    patch_h = H // h 
    patch_w = W // w 

    patches_unfolded = F.unfold(
        image_ori,
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w)
    ) 

    N = h * w
    patches = patches_unfolded.permute(0, 2, 1).reshape(B, N, C, patch_h, patch_w)
    patches = patches.squeeze(0) 

    if myway == "sobel":
        edge_sobel = torch.tensor([calculate_edge_strength_sobel(patch) for patch in patches], device=image_ori.device)
        dynamic_ratio = edge_sobel
    elif myway == "laplace":
        edge_laplace = torch.tensor([calculate_edge_strength_laplace(patch) for patch in patches], device=image_ori.device)
        dynamic_ratio = edge_laplace

    return torch.mean(dynamic_ratio), dynamic_ratio

def CDPruner(image_features, image_embeds, text_embeds, ratio=0.75):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        B, N, C = image_features.shape
        visual_token_num = int(N * ratio)
        device = image_features.device
        # index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # [CDPruner] Calculate cosine similarity
        image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
        image_normalized = image_normalized.float() # (B, N, D)
        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)
        
        # [CDPruner] Calculate query relevance
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True) # (B, N, C)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
        relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
        relevance = (-relevance).mean(dim=-1) # (B, N)
        relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)


        # [CDPruner] Construct kernel matrix
        # You can use an additional hyperparameter theta to control the influence of the relevance score.
        # theta = 0.5
        # alpha = theta / (2 * (1 - theta))
        # relevance = torch.exp(alpha * relevance) # (B, N)
        # importance = torch.exp(0.5 * importance)
        kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1) # (B, N, N)
        # kernel = similarity # (B, N, N)

        # [CDPruner] Fast MAP inference of conditional DPP
        cis = torch.zeros((visual_token_num, B, N), device=device) # (T, B, N)
        di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone() # (B, N)
        select_idx = torch.empty((visual_token_num, B), dtype=torch.long, device=device) # (T, B)
        for i in range(visual_token_num):
            j = torch.argmax(di2s, dim=-1)
            select_idx[i] = j

            eis = (kernel[torch.arange(B), j] - torch.einsum('tb,tbn->bn', cis[:i, torch.arange(B), j], cis[:i])) \
                / torch.sqrt(di2s[torch.arange(B), j]).unsqueeze(-1)
            cis[i, :, :] = eis
            di2s -= torch.square(eis)
            di2s[torch.arange(B), j] = -float('inf')
        
        select_idx = torch.sort(select_idx.t()).values # (B, T)
    #     index_masks = torch.zeros(B, N, dtype=torch.bool, device=device)
    #     index_masks.scatter_(1, select_idx, True)
    # assert index_masks.sum().item() == visual_token_num
    # return index_masks[0]
    return select_idx.squeeze(0)

def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity

def DivPrune(visual_feature_vectors, cosine_matrix=None, threshold_ratio=0.1):   
    image_feature_length = visual_feature_vectors.shape[-2]         
    threshold_terms = int(round(threshold_ratio*image_feature_length))
    # threshold_terms = int(round((threshold_ratio * image_feature_length).item()))
    if cosine_matrix is None:
        cosine_matrix = 1.0 - (pairwise_cosine_similarity(visual_feature_vectors))

    s = torch.empty(threshold_terms, dtype=torch.long, device=visual_feature_vectors.device)
    for i in range(threshold_terms):
        if i==0:
            m2 = cosine_matrix
        else:
            m2 = torch.index_select(cosine_matrix, 0, torch.index_select(s,0,torch.arange(0,i,device=cosine_matrix.device)))

        if i==0:
            scores = torch.topk(m2, 2,dim=0,largest=False).values[1,:] #for distance
        else:
            scores = torch.min(m2, dim=0).values #for distance 

        phrase_to_add_idx = torch.argmax(scores)
        s[i] = phrase_to_add_idx
    return s

def Visionzip(attn_weights, contextual_num, ratio=0.75):
    ## Dominant Visual Tokens
    cls_idx = 0
    cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
    cls_attention_sum = cls_attention.sum(dim=1) 
    dominant_num = int(cls_attention_sum.shape[-1] * ratio) - contextual_num
    topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices
    
    return topk_indices

def merge_visionzip(
    selected_prunable_subindices: torch.Tensor,
    prunable_features: torch.Tensor,
    contextual_num: int
) -> tuple[torch.Tensor, torch.Tensor]:
    device = prunable_features.device
    
    if prunable_features.dim() != 2:
        raise ValueError(f"prunable_features must be 2-dimensional (N_p, D), got {prunable_features.dim()}-dimensional")
    N_p, D = prunable_features.shape
    

    all_prunable_subindices = torch.arange(N_p, device=device)

    if selected_prunable_subindices.dim() != 1:
        selected_prunable_subindices = selected_prunable_subindices.flatten()
    
    isin_mask = torch.isin(all_prunable_subindices, selected_prunable_subindices)
    index_prop = all_prunable_subindices[~isin_mask] 
    M = index_prop.shape[0]

    if M <= contextual_num:
        target_indices_selected = index_prop 

        merged_contextual_features = torch.gather(
            prunable_features, 
            dim=0,  
            index=index_prop.unsqueeze(-1).expand(-1, D)  
        )
        return target_indices_selected, merged_contextual_features

    metric_filtered = torch.gather(
        prunable_features, 
        dim=0, 
        index=index_prop.unsqueeze(-1).expand(-1, D)  
    )

    metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True)
    metric_normalized = torch.nan_to_num(metric_normalized, 0.0)

    step = max(1, M // contextual_num)
    target_local_indices = torch.arange(0, M, step, device=device)[:contextual_num]  # [contextual_num]
    
    target_indices_selected = torch.gather(
        index_prop, 
        dim=0,  
        index=target_local_indices  
    )

    all_local_indices = torch.arange(M, device=device)  # [0~M-1]
    merge_local_mask = ~torch.isin(all_local_indices, target_local_indices)
    merge_local_indices = all_local_indices[merge_local_mask]  # [M-contextual_num]

    target_tokens = metric_normalized[target_local_indices, :]  # [contextual_num, D]
    tokens_to_merge = metric_normalized[merge_local_indices, :]  # [M-contextual_num, D]

    similarity = torch.mm(tokens_to_merge, target_tokens.transpose(0, 1))

    assign_one_hot = torch.zeros(
        merge_local_indices.shape[0], contextual_num, 
        dtype=prunable_features.dtype, device=device
    )
    max_sim_indices = similarity.argmax(dim=1).unsqueeze(-1)  
    assign_one_hot.scatter_(1, max_sim_indices, 1.0) 

    counts = assign_one_hot.sum(dim=0).clamp(min=1).unsqueeze(-1)  


    hidden_to_merge = metric_filtered[merge_local_indices, :]
    aggregated_hidden = torch.mm(assign_one_hot.transpose(0, 1), hidden_to_merge) / counts  

    target_hidden = metric_filtered[target_local_indices, :] 
    merged_contextual_features = target_hidden + aggregated_hidden  

    return target_indices_selected, merged_contextual_features

def aggregate_visionzip(x: torch.Tensor, 
              index_kept: torch.Tensor, 
              index_prop: torch.Tensor, 
              alpha: float = 0.1) -> torch.Tensor:
    B, N, C = x.shape
    device = x.device
    dtype = x.dtype

    index_kept_exp = index_kept.unsqueeze(-1).expand(B, -1, C)  # [B, K, C]
    x_kept = torch.gather(x, dim=1, index=index_kept_exp)       # [B, K, C]

    index_prop_exp = index_prop.unsqueeze(-1).expand(B, -1, C)  # [B, M, C]
    x_prop = torch.gather(x, dim=1, index=index_prop_exp)       # [B, M, C]

    norm_prop = x_prop.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, M, 1]
    x_prop_normalized = x_prop / norm_prop
    x_prop_normalized = torch.nan_to_num(x_prop_normalized, 0.0)

    norm_kept = x_kept.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, K, 1]
    x_kept_normalized = x_kept / norm_kept
    x_kept_normalized = torch.nan_to_num(x_kept_normalized, 0.0)

    # [B, M, K]
    similarity = torch.bmm(x_prop_normalized, x_kept_normalized.transpose(1, 2))

    max_sim_indices = similarity.argmax(dim=2)  # [B, M]
    assign_one_hot = torch.zeros(B, max_sim_indices.shape[1], x_kept.shape[1], 
                                 dtype=dtype, device=device)  # [B, M, K]
    assign_one_hot.scatter_(2, max_sim_indices.unsqueeze(-1), 1.0)

    counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)  # [B, K, 1]

    aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), x_prop) / counts  # [B, K, C]

    x_merged = x_kept + alpha * aggregated_hidden  # [B, K, C]

    return x_merged

def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, :, idx]

def aggregate_sparsevlm(x: torch.Tensor, 
                        index_kept: torch.Tensor, 
                        index_prop: torch.Tensor, 
                        alpha: float = 0.1) -> torch.Tensor:

    B, N, C = x.shape
    K = index_kept.shape[1]  
    M = index_prop.shape[1]  
    device = x.device
    dtype = x.dtype


    index_kept_exp = index_kept.unsqueeze(-1).expand(B, K, C)  # [B, K, C]
    x_kept = torch.gather(x, dim=1, index=index_kept_exp)       # [B, K, C]

    index_prop_exp = index_prop.unsqueeze(-1).expand(B, M, C)   # [B, M, C]
    x_prop = torch.gather(x, dim=1, index=index_prop_exp)       # [B, M, C]

    # x_prop [B, M, 1, C], x_kept [B, 1, K, C]
    x_prop_exp = ein.rearrange(x_prop, "b m c -> b m () c")
    x_kept_exp = ein.rearrange(x_kept, "b k c -> b () k c")
    
    # [B, M, K]
    distance = (x_prop_exp - x_kept_exp).norm(dim=-1, p=2)
    dist_matrix = distance / (C ** 0.5)

    idx_cluster = dist_matrix.argmin(dim=2)  # [B, M]

    idx_batch = torch.arange(B, device=device)[:, None].expand(B, K)  # [B, K]
    idx_tmp = torch.arange(K, device=device)[None, :].expand(B, K)    # [B, K]
    for b in range(B):
        keep_in_prop = torch.isin(index_prop[b], index_kept[b])
        if keep_in_prop.any():
            prop_idx = torch.where(keep_in_prop)[0]
            keep_idx = torch.where(torch.isin(index_kept[b], index_prop[b][prop_idx]))[0]
            idx_cluster[b, prop_idx] = keep_idx

    # [B, M]
    idx_global_cluster = idx_cluster + idx_batch[:, 0:1] * K

    token_weight = torch.ones(B, M, 1, device=device, dtype=dtype)  # [B, M, 1]

    all_weight = torch.zeros(B * K, 1, device=device, dtype=dtype)
    all_weight.index_add_(
        dim=0,
        index=idx_global_cluster.reshape(B * M),
        source=token_weight.reshape(B * M, 1)
    )
    all_weight = all_weight.clamp(min=1e-6) 

    norm_weight = token_weight / all_weight[idx_global_cluster]  # [B, M, 1]

    x_prop_weighted = x_prop * norm_weight  # [B, M, C]

    x_aggregated = torch.zeros(B * K, C, device=device, dtype=dtype)
    x_aggregated.index_add_(
        dim=0,
        index=idx_global_cluster.reshape(B * M),
        source=x_prop_weighted.reshape(B * M, C)
    )
    x_aggregated = x_aggregated.reshape(B, K, C)  # [B, K, C]

    x_merged = x_kept + alpha * x_aggregated  # [B, K, C]

    return x_merged

def _create_distance_penalty_matrix(
    num_patches: int,
    distance_threshold: float,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    
    side_len = int(math.sqrt(num_patches))
    if side_len * side_len != num_patches:
        side_len = math.ceil(math.sqrt(num_patches))
        coords = torch.meshgrid(
            torch.arange(side_len, device=device, dtype=torch.int64),
            torch.arange(side_len, device=device, dtype=torch.int64),
            indexing="ij"
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 2)[:num_patches]
    else:
        coords = torch.meshgrid(
            torch.arange(side_len, device=device, dtype=torch.int64),
            torch.arange(side_len, device=device, dtype=torch.int64),
            indexing="ij"
        )
        coords = torch.stack(coords, dim=-1).reshape(-1, 2)

    coords_float = coords.to(dtype=torch.float32)
    coords_exp = coords_float.unsqueeze(1)
    
    dist_matrix = torch.norm(coords_exp - coords_float, dim=-1, p=2)

    dist_penalty = (dist_matrix <= distance_threshold).to(dtype=dtype)

    return dist_penalty

def aggregate_nuwa(
    x: torch.Tensor,
    index_kept: torch.Tensor,
    index_prop: torch.Tensor,
    alpha: float = 0.1,
    distance: int = 280,
    PENALTY_THRESHOLD_PERCENTILE: float = 0.55,
    min_layers: int = 24,
    L2NORM: bool = True,
    clip_hidden_states: list = None  
) -> torch.Tensor:

    B, N, C = x.shape
    K = index_kept.shape[1] 
    M = index_prop.shape[1] 
    device = x.device
    dtype = x.dtype

    if clip_hidden_states is None:
        clip_hidden_states = [torch.cat([torch.zeros(B, 1, C, device=device), x], dim=1) for _ in range(24)]
    if len(clip_hidden_states) < min_layers:
        raise ValueError(f"CLIP must have at least {min_layers} layers (got {len(clip_hidden_states)}).")


    index_kept_exp = index_kept.unsqueeze(-1).expand(B, K, C)  
    x_kept = torch.gather(x, dim=1, index=index_kept_exp)       

    index_prop_exp = index_prop.unsqueeze(-1).expand(B, M, C)  
    x_prop = torch.gather(x, dim=1, index=index_prop_exp)   

    stacked_hs = torch.stack(clip_hidden_states[16:25], dim=0)  
    avg_hs = stacked_hs.mean(dim=0)[:, 1:, :]  
    avg_hs_kept = torch.gather(avg_hs, dim=1, index=index_kept_exp)
    avg_hs_prop = torch.gather(avg_hs, dim=1, index=index_prop_exp)

    avg_hs_kept_norm = F.normalize(avg_hs_kept, p=2, dim=-1)    # [B, K, C]
    avg_hs_prop_norm = F.normalize(avg_hs_prop, p=2, dim=-1)    # [B, M, C]

    sim_matrix = torch.bmm(avg_hs_kept_norm, avg_hs_prop_norm.transpose(1, 2))

    dist_penalty = _create_distance_penalty_matrix(
        num_patches=N,
        distance_threshold=math.sqrt(distance),
        device=device,
        dtype=dtype
    )  # [N, N]

    dist_penalty_batch = []
    for b in range(B):
        penalty_kept = dist_penalty[index_kept[b]]  # [K, N]
        penalty_kept_prop = penalty_kept[:, index_prop[b]]  # [K, M]
        dist_penalty_batch.append(penalty_kept_prop)
    dist_penalty_batch = torch.stack(dist_penalty_batch, dim=0)  # [B, K, M]

    aggregation_weights = F.relu(sim_matrix) * dist_penalty_batch  # [B, K, M]

    aggregation_weights_norm = aggregation_weights / (
        aggregation_weights.sum(dim=-1, keepdim=True) + 1e-8
    )  # [B, K, M]

    for b in range(B):
        keep_in_prop = torch.isin(index_prop[b], index_kept[b])
        if keep_in_prop.any():
            prop_idx = torch.where(keep_in_prop)[0]  
            keep_idx = torch.where(torch.isin(index_kept[b], index_prop[b][prop_idx]))[0]  
            aggregation_weights_norm[b, keep_idx, prop_idx] = 1.0

    if L2NORM:
        token_l2_norms = torch.linalg.norm(x, ord=2, dim=-1)  # [B, N]
        benchmark_l2_norms = torch.gather(token_l2_norms, dim=1, index=index_kept)  # [B, K]
        norm_threshold = torch.quantile(
            benchmark_l2_norms.float(), PENALTY_THRESHOLD_PERCENTILE, dim=1, keepdim=True
        ).to(dtype)  # [B, 1]
        is_high_norm_token = benchmark_l2_norms >= norm_threshold  # [B, K]
    else:
        is_high_norm_token = None

    x_prop = x_prop.to(aggregation_weights_norm.dtype)
    aggregated_tokens = torch.bmm(aggregation_weights_norm, x_prop)  # [B, K, C]

    if L2NORM and is_high_norm_token is not None:
        mask_expanded = is_high_norm_token.unsqueeze(-1).expand_as(aggregated_tokens)  # [B, K, C]
        aggregated_tokens = torch.where(
            mask_expanded,
            x_kept,  
            aggregated_tokens  
        )

    x_merged = x_kept + alpha * aggregated_tokens  # [B, K, C]

    return x_merged

def ssd_prune(prunable_features, importance, h, w, ratio=0.5):

    prunable_features = prunable_features / prunable_features.norm(dim=-1, keepdim=True)
    device = prunable_features.device
    N = prunable_features.shape[0]
    assert N == h * w, f"feature {N} mismatch h*w={h*w}"
    visual_token_num = int(N * ratio)

    # SSD Algorithm
    select_idx = torch.empty((visual_token_num,1), dtype=torch.long, device=device)
    gamma = 1.0
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    i0 = torch.argmax(importance)
    select_idx[0] = i0
    selected_mask[i0] = True  # 标记为已选中
    # V = gamma * prunable_features[i0].norm(dim=-1)  # V仅需计算一次

    for t in range(1, visual_token_num):
        # 直接通过掩码获取未选中的候选token（替代原列表推导式，O(1)效率）
        candidates = torch.where(~selected_mask)[0]  # shape: (k,)，k = N - t
        if len(candidates) == 0:
            break  # 极端情况：候选集为空（一般不会触发）
        
        # 上一轮选中的token特征
        last_selected = select_idx[t-1, 0]  # 取第0个batch的上一轮结果
        v_last = prunable_features[last_selected].unsqueeze(0)  # shape: (1, d)
        
        # 向量化计算所有候选token的投影（替代原循环，批量操作）
        v_candidates = prunable_features[candidates]  # shape: (k, d)
        # 计算投影：(k,d)与(1,d)的矩阵乘法 -> (k,1)，再乘以v_last得到(k,d)
        projection = (torch.matmul(v_candidates, v_last.T) / torch.matmul(v_last, v_last.T)) * v_last
        # 更新候选token的embeddings（批量操作）
        prunable_features[candidates] = v_candidates - projection.squeeze(1)  # squeeze掉维度1
        
        # 向量化计算所有候选的分数（替代原循环）
        scores = importance[candidates] + prunable_features[candidates].norm(dim=-1) #+ (torch.rand(len(candidates), device=importance.device) * 0.0002 - 0.0001)  # shape: (k,)
        
        # 选最高分的候选
        best_idx = torch.argmax(scores)
        best_j = candidates[best_idx]
        # print(best_j)
        select_idx[t] = best_j
        selected_mask[best_j] = True  # 标记为已选中
        # V = V * prunable_features[best_j].norm(dim=-1)

    # 后续处理保持不变
    # print(select_idx)
    keep_indices = select_idx.clone().detach().sort().values
    keep_indices = keep_indices.squeeze(1)

    return keep_indices

def rank_prune_simple(prunable_features, importance, h, w, ratio=0.75):
    N = prunable_features.shape[0]
    # assert N == h * w, f"feature {N} mismatch h*w={h*w}"
    
    num_to_keep = int(N * ratio)
    
    _, keep_indices = torch.topk(importance, num_to_keep)
    
    keep_indices = keep_indices.sort().values
    
    return keep_indices

def is_boundary(idx, h, w):
    row = idx // w
    col = idx % w
    return row == 0 or row == h-1 or col == 0 or col == w-1

def RandomPrune(prunable_features, ratio=0.5):

    total_elements = prunable_features.shape[0]
    device = prunable_features.device
    num_to_keep = int(total_elements * ratio)
    num_to_keep = max(num_to_keep, 1)

    all_indices = torch.randperm(total_elements, device=device)
    keep_index = all_indices[:num_to_keep]

    return keep_index

def get_graph(x: torch.Tensor):
    B, N, C = x.shape

    x_normed = x / x.norm(dim=-1, keepdim=True)
    x_cossim = x_normed @ x_normed.transpose(-1, -2)

    # ==========================================
    x_cossim = x_cossim[0]
    diag = torch.diag(x_cossim)
    diag = torch.diag_embed(diag)
    graph = x_cossim - diag
    graph = torch.where(graph>=0.0, graph, 0.0)
    graph = graph.unsqueeze(0).expand(B,-1,-1)
    # ==========================================
            
    # Symmetrically normalize the graph
    degree = graph.sum(-1) # B, N
    degree = torch.diag_embed(degree**(-1/2))
    graph = degree @ graph @ degree

    return graph

def aggregate(x, index_kept, index_prop, alpha = 0.1):

    B, N, C = x.shape
    weight = get_graph(x)

    x_kept = x.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,C).to(x.device)) # B, N-1-num_prop, C
    x_prop = x.gather(dim=1, index=index_prop.unsqueeze(-1).expand(-1,-1,C).to(x.device)) # B, num_prop, C

    weight = weight.gather(dim=1, index=index_kept.unsqueeze(-1).expand(-1,-1,N).to(weight.device)) # B, N-1-num_prop, N-1
    weight_prop = weight.gather(dim=2, index=index_prop.unsqueeze(1).expand(-1,weight.shape[1],-1).to(weight.device)) # B, N-1-num_prop, num_prop
    # weight = weight.gather(dim=2, index=index_kept.unsqueeze(1).expand(-1,weight.shape[1],-1).to(weight.device)) # B, N-1-num_prop, N-1-num_prop
    weight_prop = weight_prop.to(x_prop.dtype)
    x_prop = weight_prop @ x_prop # B, N-1-num_prop, C
    x_kept = x_kept + alpha * x_prop # B, N-1-num_prop, C

    x = x_kept

    return x

def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)

def log_optimal_transport(scores: torch.Tensor, iters: int) -> torch.Tensor:
    alpha = torch.nn.Parameter(torch.tensor(0.2, device=scores.device))

    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z[:, :-1, :-1]

def aggregate_ot(
    x: torch.Tensor,
    index_kept: torch.Tensor,
    index_prop: torch.Tensor,
    alpha: float = 0.1,
) -> torch.Tensor:

    B, N_total, C = x.shape
    device = x.device
    
    idx_kept_expand = index_kept.unsqueeze(-1).expand(B, -1, C)
    x_kept = torch.gather(x, dim=1, index=idx_kept_expand)  # [B, M, C]
    
    idx_prop_expand = index_prop.unsqueeze(-1).expand(B, -1, C)
    x_prop = torch.gather(x, dim=1, index=idx_prop_expand)  # [B, P, C]
    
    x_kept_norm = F.normalize(x_kept, p=2, dim=-1)
    x_prop_norm = F.normalize(x_prop, p=2, dim=-1)
    sim_matrix = torch.bmm(x_kept_norm, x_prop_norm.transpose(1, 2))
    
    transport_log = log_optimal_transport(sim_matrix, iters=100)
    transport_matrix = transport_log.exp()
    
    x_prop_fused = torch.bmm(transport_matrix, x_prop)
    
    x_kept_fused = x_kept + alpha * x_prop_fused
    
    return x_kept_fused

def calculate_edge_strength_sobel(patch, threshold=0.2):

    gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]  
    gray = gray.unsqueeze(0).unsqueeze(0)  

    sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32, device=patch.device)
    sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32, device=patch.device)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_pixels_mask = (grad_magnitude > threshold)
    edge_pixel_count = torch.sum(edge_pixels_mask).item()
    total_pixel_count = grad_magnitude.numel()  
    edge_ratio = edge_pixel_count / total_pixel_count
    
    return edge_ratio

def calculate_edge_strength_laplace(patch, threshold=0.2):
    gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
    gray = gray.unsqueeze(0).unsqueeze(0)  

    laplace_kernel = torch.tensor([[[[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]]]], dtype=torch.float32, device=patch.device)
    
    # laplace_kernel = torch.tensor([[[[1, 1, 1],
    #                                  [1, -8, 1],
    #                                  [1, 1, 1]]]], dtype=torch.float32, device=patch.device)

    laplace_response = F.conv2d(gray, laplace_kernel, padding=1)
    edge_strength = torch.abs(laplace_response)
    edge_pixels_mask = (edge_strength > threshold)  
    edge_pixel_count = torch.sum(edge_pixels_mask).item()
    total_pixel_count = edge_strength.numel()  
    
    edge_ratio = edge_pixel_count / total_pixel_count
    
    return edge_ratio

def merge_consecutive_branch_indices(selected_indices, branch_indices_set):

    selected_list = selected_indices.tolist()
    final_kept = []
    prev_was_branch = False

    for idx in selected_list:
        is_branch = idx in branch_indices_set
        
        if is_branch and prev_was_branch:
            continue
        
        final_kept.append(idx)
        prev_was_branch = is_branch

    return torch.tensor(final_kept, dtype=torch.long, device=selected_indices.device)