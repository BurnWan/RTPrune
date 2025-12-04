import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap  

def draw_line_chart(output_path, importance, h, w):
    base_size = h * w
    sim_map = importance.reshape(base_size, base_size).unsqueeze(0).unsqueeze(0)
    sim_map = F.interpolate(sim_map, size=(base_size, base_size), mode='nearest')  # 插值到base_size×base_size
    
    sim_map = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    
    sim_map_2d = sim_map.cpu().squeeze().detach().numpy()  # 形状: (base_size, base_size)

    plt.figure(figsize=(8, 8))  # 控制图的尺寸
    colors = [(1, 1, 1), (0, 0, 1)]
    white_to_blue = LinearSegmentedColormap.from_list("white_to_blue", colors, N=256)
    
    im = plt.imshow(
        sim_map_2d, 
        cmap=white_to_blue,  # 核心：与示例颜色匹配的colormap
        vmin=0, vmax=1,   # 固定取值范围
        extent=[0, base_size-1, base_size-1, 0]  # 让y轴0在顶部（匹配示例坐标轴方向）
    )
    
    
    # plt.xlabel('X Index')
    # plt.ylabel('Y Index')
    plt.xticks(np.arange(0, base_size, 50))  # 间隔10个刻度（示例是0/10/20...）
    plt.yticks(np.arange(0, base_size, 50))

    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()  # 关闭画布释放资源

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

def CDPruner(image_features, importance, ratio=0.9):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=False):
        B, N, C = image_features.shape
        visual_token_num = int(N * ratio)
        device = image_features.device
        index_masks = torch.ones(B, N, dtype=torch.bool, device=device)
        
        # [CDPruner] Calculate cosine similarity
        image_normalized = image_features / image_features.norm(dim=-1, keepdim=True) # (B, N, D)
        image_normalized = image_normalized.float() # (B, N, D)
        similarity = torch.matmul(image_normalized, image_normalized.transpose(1, 2)) # (B, N, N)
        
        # text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True) # (M, C)
        # relevance = torch.matmul(image_embeds, text_embeds.t()) # (B, N, M)
        # relevance = (-relevance).mean(dim=-1) # (B, N)
        # relevance = (relevance - relevance.min() + 1e-6) / (relevance.max() - relevance.min()) # (B, N)


        # [CDPruner] Construct kernel matrix
        # You can use an additional hyperparameter theta to control the influence of the relevance score.
        # theta = 0.5
        # alpha = theta / (2 * (1 - theta))
        # relevance = torch.exp(alpha * relevance) # (B, N)
        # importance = torch.exp(0.5 * importance)
        kernel = importance.unsqueeze(2) * similarity * importance.unsqueeze(1) # (B, N, N)
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

# divprune
def pairwise_cosine_similarity(matrix):
    norm_matrix = matrix / matrix.norm(dim=1, keepdim=True)
    cosine_similarity = torch.mm(norm_matrix, norm_matrix.t())
    return cosine_similarity

def DivPrune(visual_feature_vectors, cosine_matrix=None, threshold_ratio=0.1):   
    image_feature_length = visual_feature_vectors.shape[-2]         
    threshold_terms = int(round(threshold_ratio*image_feature_length))
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

def ssd_prune(prunable_features, importance, h, w, ratio=0.5):
    # 计算每个token的自身信息量分数（L2范数，值越大信息量越高）
    # info_scores = torch.norm(prunable_features, dim=1)
    # print(info_scores)

    prunable_features = prunable_features / prunable_features.norm(dim=-1, keepdim=True)
    # if ratio == 0.5:
    #     m = 4
    #     n = 2
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
    assert N == h * w, f"feature {N} mismatch h*w={h*w}"
    
    num_to_keep = int(N * ratio)
    
    _, keep_indices = torch.topk(importance, num_to_keep)
    
    keep_indices = keep_indices.sort().values
    
    return keep_indices

def spatial_aware_prune(prunable_features, importance, h, w, ratio=0.75, spatial_ratio=0.15):
    N = prunable_features.shape[0]
    assert N == h * w, f"feature {N} mismatch h*w={h*w}"
    device = prunable_features.device
    
    # 1. 计算各阶段需要保留的token数量
    num_total_keep = int(N * ratio)
    num_spatial_keep = int(N * spatial_ratio)
    num_spatial_keep = max(num_spatial_keep, 4)  # 至少保留4个角点（边界的基础）
    num_importance_keep = max(num_total_keep - num_spatial_keep, 0)
    
    # 2. 空间采样：优先覆盖最外边一圈，再填充中间
    spatial_indices = []
    
    # 2.1 强制保留四个角点（边界的关键节点）
    corners = [
        0 * w + 0,                  # 左上角 (0,0)
        0 * w + (w-1),              # 右上角 (0, w-1)
        (h-1) * w + 0,              # 左下角 (h-1, 0)
        (h-1) * w + (w-1)           # 右下角 (h-1, w-1)
    ]
    spatial_indices.extend(corners)
    
    # 2.2 计算剩余需要采样的点（边界+中间）
    remaining_spatial = num_spatial_keep - 4
    if remaining_spatial <= 0:
        # 若总需求≤4，仅保留四角（已覆盖边界关键位置）
        spatial_indices = corners
    else:
        # 2.2.1 边界采样：覆盖最外边一圈的四条边（排除已保留的四角）
        # 四条边的定义：上（行0，列1~w-2）、下（行h-1，列1~w-2）、左（列0，行1~h-2）、右（列w-1，行1~h-2）
        boundary_points = []
        
        # 计算每条边需要的采样点数（按边的长度比例分配）
        # 上/下边长度：w-2（排除2个角），左/右边长度：h-2（排除2个角）
        top_bottom_len = w - 2
        left_right_len = h - 2
        total_boundary_len = 2 * (top_bottom_len + left_right_len)  # 边界总长度（不含四角）
        
        # 分配边界采样点（占剩余空间采样数的70%，确保边界覆盖充分）
        num_boundary = int(remaining_spatial * 0.7)
        num_inner = remaining_spatial - num_boundary
        
        # 上边缘采样（行0，列1~w-2）
        if top_bottom_len > 0 and num_boundary > 0:
            step = max(1, top_bottom_len // max(1, num_boundary // 4))  # 均匀步长
            top_points = [0 * w + c for c in range(1, w-1, step)]
            boundary_points.extend(top_points)
        
        # 下边缘采样（行h-1，列1~w-2）
        if top_bottom_len > 0 and num_boundary > 0:
            step = max(1, top_bottom_len // max(1, num_boundary // 4))
            bottom_points = [(h-1) * w + c for c in range(1, w-1, step)]
            boundary_points.extend(bottom_points)
        
        # 左边缘采样（列0，行1~h-2）
        if left_right_len > 0 and num_boundary > 0:
            step = max(1, left_right_len // max(1, num_boundary // 4))
            left_points = [r * w + 0 for r in range(1, h-1, step)]
            boundary_points.extend(left_points)
        
        # 右边缘采样（列w-1，行1~h-2）
        if left_right_len > 0 and num_boundary > 0:
            step = max(1, left_right_len // max(1, num_boundary // 4))
            right_points = [r * w + (w-1) for r in range(1, h-1, step)]
            boundary_points.extend(right_points)
        
        # 2.2.2 中间区域采样（排除最外边一圈的内部区域）
        inner_points = []
        if h >= 3 and w >= 3 and num_inner > 0:
            # 中间区域范围：行1~h-2，列1~w-2（完全在边界内部）
            inner_h, inner_w = h-2, w-2
            # 网格划分确保中间点均匀分布
            grid_h = max(1, int(np.sqrt(num_inner * inner_h / inner_w)))
            grid_w = max(1, int(np.sqrt(num_inner * inner_w / inner_h)))
            step_h = (inner_h + grid_h - 1) // grid_h  # 向上取整，避免遗漏
            step_w = (inner_w + grid_w - 1) // grid_w
            
            for i in range(grid_h):
                for j in range(grid_w):
                    r = 1 + i * step_h  # 行偏移1，避开上边界
                    c = 1 + j * step_w  # 列偏移1，避开左边界
                    # 确保不超出中间区域（避免触及下、右边界）
                    if r < h-1 and c < w-1:
                        inner_points.append(r * w + c)
        
        # 2.2.3 合并并调整数量（确保不超过需求，优先保留边界）
        spatial_indices.extend(boundary_points)
        spatial_indices.extend(inner_points)
        # 去重（避免边界与中间重叠，或边界点重复）
        spatial_indices = list(set(spatial_indices))
        
        # 若总数超过需求，优先保留四角和边界，再截断中间点
        if len(spatial_indices) > num_spatial_keep:
            # 分离四角、边界、中间点
            non_corner_boundary = [idx for idx in spatial_indices if idx not in corners and is_boundary(idx, h, w)]
            inner_only = [idx for idx in spatial_indices if not is_boundary(idx, h, w) and idx not in corners]
            # 优先保留：四角 → 边界 → 中间
            keep_list = corners + non_corner_boundary + inner_only
            # 截断到需要的数量
            spatial_indices = keep_list[:num_spatial_keep]
    
    # 转换为张量并确保设备一致
    spatial_indices = torch.tensor(spatial_indices, dtype=torch.long, device=device)
    # 再次去重（避免极端情况）
    spatial_indices = torch.unique(spatial_indices)
    
    # 3. 从剩余token中按importance选择
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    selected_mask[spatial_indices] = True
    remaining_indices = torch.where(~selected_mask)[0]
    
    if num_importance_keep > 0 and len(remaining_indices) > 0:
        remaining_importance = importance[remaining_indices]
        _, top_remaining_idx = torch.topk(remaining_importance, min(num_importance_keep, len(remaining_indices)))
        importance_indices = remaining_indices[top_remaining_idx]
    else:
        importance_indices = torch.tensor([], dtype=torch.long, device=device)
    
    # 4. 合并结果并排序
    keep_indices = torch.cat([spatial_indices, importance_indices])
    keep_indices = keep_indices[:num_total_keep]  # 确保不超过总需求
    keep_indices = keep_indices.sort().values
    
    return keep_indices

# 辅助函数：判断一个索引是否在最外边一圈
def is_boundary(idx, h, w):
    row = idx // w
    col = idx % w
    # 边界条件：行是0或h-1，或列是0或w-1
    return row == 0 or row == h-1 or col == 0 or col == w-1

def RowPrune(prunable_features, importance, h, w, ratio=0.5):
    N = prunable_features.shape[0]
    device = prunable_features.device
    assert N == h * w, f"feature {N} mismatch h*w={h*w}"

    # 1. 将 1D 的 importance 分数重塑为 2D 矩阵 (h, w)
    weights_reshaped = importance.view(h, w)

    # 2. 计算每行和每列的重要性总和
    row_sums = weights_reshaped.sum(dim=1)  # 形状: (h,)
    col_sums = weights_reshaped.sum(dim=0)  # 形状: (w,)

    # 3. 确定哪些行和列是有效的（和不为0）
    valid_rows = (row_sums != 0)  # 形状: (h,)，布尔张量
    valid_cols = (col_sums != 0)  # 形状: (w,)，布尔张量

    # 4. 生成一个 2D 掩码，标记哪些 (行, 列) 位置是有效的
    # 一个位置有效，当且仅当它所在的行和列都有效
    valid_mask_2d = valid_rows.unsqueeze(1) & valid_cols.unsqueeze(0)  # 形状: (h, w)

    # 5. 将 2D 掩码展平为 1D，并找到所有有效位置的下标
    keep_index = torch.where(valid_mask_2d.view(-1))[0]  # 形状: (num_to_keep,)

    # N, C = prunable_features.shape
    # device = prunable_features.device
    # weights_reshaped = importance.view(h, w)
    # row_importance = weights_reshaped.mean(dim=1)

    # num_rows_to_keep = int(h * ratio)
    # _, top_row_indices = torch.topk(row_importance, num_rows_to_keep, dim=0)
    # # print(top_row_indices)
    # keep_index_list = []
    # for row_idx in top_row_indices:
    #     start_idx = row_idx * w
    #     row_token_indices = torch.arange(start_idx, start_idx + w, device=device)
    #     keep_index_list.append(row_token_indices)
    # keep_index = torch.cat(keep_index_list)

    return keep_index

def RandomPrune(prunable_features, ratio=0.5):

    total_elements = prunable_features.shape[0]  # 等价于 N * C
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

def filter_patches_by_info(image_ori, h, w, myway="sobel", alpha=1.0):
    B, C, H, W = image_ori.shape
    patch_h = H // h  # 每个Patch的高度
    patch_w = W // w  # 每个Patch的宽度

    # 使用unfold拆分Patch (B, C*patch_h*patch_w, N)，N=h×w为Patch总数
    patches_unfolded = F.unfold(
        image_ori,
        kernel_size=(patch_h, patch_w),
        stride=(patch_h, patch_w)
    )  # 输出形状：(1, 3*patch_h*patch_w, h*w)

    # 调整为 (B, N, C, patch_h, patch_w)，方便后续处理每个Patch
    N = h * w
    patches = patches_unfolded.permute(0, 2, 1).reshape(B, N, C, patch_h, patch_w)
    patches = patches.squeeze(0)  # 去掉batch维度，形状：(N, 3, patch_h, patch_w)，N=h×w

    # all_entropies = torch.tensor([calculate_entropy(patch) for patch in patches], device=image_ori.device)
    # all_density = torch.tensor([calculate_patch_density(patch) for patch in patches], device=image_ori.device)
    edge_sobel = torch.tensor([calculate_edge_strength_sobel(patch) for patch in patches], device=image_ori.device)
    edge_laplace = torch.tensor([calculate_edge_strength_laplace(patch) for patch in patches], device=image_ori.device)

    # entropy = (all_entropies - all_entropies.min()) / (all_entropies.max() - all_entropies.min() + 1e-8)
    # density = (all_density - all_density.min()) / (all_density.max() - all_density.min() + 1e-8)
    # norm_edge = (all_edges - all_edges.min()) / (all_edges.max() - all_edges.min() + 1e-8)
    if myway == "sobel":
        importance = edge_sobel # + entropy
    elif myway == "laplace":
        importance = edge_laplace

    return importance

def calculate_entropy(patch):
    """计算单个Patch的信息熵（RGB三通道平均）"""
    # patch形状：(3, patch_h, patch_w)，先转换为 (patch_h, patch_w, 3) 并归一化到[0,255]
    patch_np = patch.permute(1, 2, 0).detach().to(torch.float).cpu().numpy()
    patch_np = (patch_np + 1) / 2  # 归一化到[0,1]
    patch_np = (patch_np * 255).astype(np.uint8)  # 转换为0-255整数

    entropy = 0.0
    for c in range(3):  # 每个通道单独计算熵，再求平均
        channel = patch_np[..., c].flatten()
        # 计算直方图（256个bins，对应0-255）
        hist, _ = np.histogram(channel, bins=256, range=(0, 255), density=True)
        # 过滤掉概率为0的项，避免log2(0)
        hist = hist[hist > 0]
        # 信息熵公式：H = -sum(p * log2(p))
        channel_entropy = -np.sum(hist * np.log2(hist))
        entropy += channel_entropy / 3  # 三通道平均
    return entropy

def calculate_patch_density(patch, method="variance", bg_threshold=240):
    # 1. Patch 预处理：转为灰度图 + 还原到 0-255 像素值
    # 转为灰度图：RGB -> Gray（加权平均）
    gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]  # (patch_h, patch_w)
    # 还原到 0-255（假设输入 patch 已归一化到 [0,1]）
    gray_255 = gray.detach().to(torch.float).cpu().numpy() * 255
    gray_255 = gray_255.astype(np.uint8)  # 避免浮点误差

    # 2. 计算密集度

    total_pixels = gray_255.size  # Patch 总像素数
    # 背景像素：接近全白（> bg_threshold）或全黑（< 15，可根据实际调整）
    bg_pixels = np.sum((gray_255 > bg_threshold))
    density = 1 - (bg_pixels / total_pixels)  # 1 - 稀疏度

    return density

def calculate_edge_strength_sobel(patch, threshold=0.15):
    """计算单个Patch的边缘强度（Sobel算子）"""
    # 1. 转换为灰度图 (3, H, W) -> (1, H, W)
    gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]  # 加权平均转灰度
    gray = gray.unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, patch_h, patch_w)，适配conv2d

    # 2. Sobel算子（x和y方向）
    sobel_x = torch.tensor([[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], dtype=torch.float32, device=patch.device)
    sobel_y = torch.tensor([[[[1, 2, 1], [0, 0, 0], [-1, -2, -1]]]], dtype=torch.float32, device=patch.device)

    # 3. 计算x和y方向梯度
    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # 4. 边缘强度：梯度的L2范数（或绝对值和），取平均值作为Patch的边缘强度
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    edge_pixels_mask = (grad_magnitude > threshold)
    edge_pixel_count = torch.sum(edge_pixels_mask).item()
    total_pixel_count = grad_magnitude.numel()  # 获取总像素数
    edge_ratio = edge_pixel_count / total_pixel_count
    
    return edge_ratio

def calculate_edge_strength_laplace(patch, threshold=0.1):
    gray = 0.299 * patch[0] + 0.587 * patch[1] + 0.114 * patch[2]
    gray = gray.unsqueeze(0).unsqueeze(0)  # 形状：(1, 1, patch_h, patch_w)

    # laplace_kernel = torch.tensor([[[[0, 1, 0],
    #                                  [1, -4, 1],
    #                                  [0, 1, 0]]]], dtype=torch.float32, device=patch.device)
    
    # 另一个常用的 Laplace 核（包含对角线方向）
    laplace_kernel = torch.tensor([[[[1, 1, 1],
                                     [1, -8, 1],
                                     [1, 1, 1]]]], dtype=torch.float32, device=patch.device)

    laplace_response = F.conv2d(gray, laplace_kernel, padding=1)
    edge_strength = torch.abs(laplace_response)
    edge_pixels_mask = (edge_strength > threshold)  
    edge_pixel_count = torch.sum(edge_pixels_mask).item()
    total_pixel_count = edge_strength.numel()  # 获取元素总数
    
    edge_ratio = edge_pixel_count / total_pixel_count
    
    return edge_ratio

def merge_consecutive_branch_indices(selected_indices, branch_indices_set):

    selected_list = selected_indices.tolist()
    final_kept = []
    prev_was_branch = False

    for idx in selected_list:
        # 检查当前索引是否是分行位
        is_branch = idx in branch_indices_set
        
        # 如果当前是分行位，并且前一个也是分行位，则跳过当前分行位
        if is_branch and prev_was_branch:
            continue
        
        # 否则，保留当前索引
        final_kept.append(idx)
        prev_was_branch = is_branch

    return torch.tensor(final_kept, dtype=torch.long, device=selected_indices.device)