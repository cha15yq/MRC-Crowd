import torch


def den2cls(gt_den_map, label_bd):
    B, C, H, W = gt_den_map.shape
    d_map = gt_den_map.flatten(1).unsqueeze(2)
    label_bd = label_bd.unsqueeze(0).unsqueeze(0)
    cls_map = ((d_map - label_bd) > 0).sum(dim=-1)
    cls_map = cls_map.reshape(B, H, W)
    return cls_map


def cls2den(pred_cls_map, indices_proxy):
    input_size = pred_cls_map.size()
    pre_cls = pred_cls_map.reshape(-1).cpu()
    pre_counts = torch.index_select(indices_proxy, 0, pre_cls.cpu().type(torch.LongTensor))
    pre_counts = pre_counts.reshape(input_size)
    return pre_counts