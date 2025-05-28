def yellow_mask(images, r_min=0.75, r_max=1.25, b_max=0.25):
    is_batch = images.ndim == 4
    if not is_batch:
        images = images.unsqueeze(0)
    
    R = images[:, 0, :, :]
    G = images[:, 1, :, :]
    B = images[:, 2, :, :]
    
    ratio = R / (G + 1e-6)
    mask = (ratio > r_min) & (ratio < r_max) & (B < b_max)
    
    return mask.float().squeeze(0) if not is_batch else mask.unsqueeze(1).float()