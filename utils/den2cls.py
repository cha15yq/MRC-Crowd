def den2cls(gt_den_map, label_bd):
    B, C, H, W = gt_den_map.shape
    d_map = gt_den_map.flatten(1).unsqueeze(2)
    label_bd = label_bd.unsqueeze(0).unsqueeze(0)
    cls_map = ((d_map - label_bd) > 0).sum(dim=-1)
    cls_map = cls_map.reshape(B, H, W)
    return cls_map

