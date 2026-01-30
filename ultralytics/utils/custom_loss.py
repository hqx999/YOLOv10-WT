import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleDetectLoss(nn.Module):
    def __init__(self, nc=1, box_weight=0.05, cls_weight=0.5):
        super().__init__()
        self.nc = nc
        self.box_weight = box_weight
        self.cls_weight = cls_weight

        self.bce_cls = nn.BCEWithLogitsLoss()
        self.iou_loss = self._ciou_loss

    def forward(self, preds, targets):
        """
        preds: list of [B, C+4, H, W] tensors
        targets: dict with keys ['batch_idx', 'cls', 'bboxes'] or preprocessed GT
        """
        device = preds[0].device
        cls_loss = torch.tensor(0.0, device=device)
        box_loss = torch.tensor(0.0, device=device)

        # Placeholder for match logic, simplified for demo
        for pred in preds:
            B, _, H, W = pred.shape
            pred = pred.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C+4]
            pred = pred.view(B, -1, self.nc + 4)          # [B, HW, C+4]

            pred_cls = pred[..., :self.nc]
            pred_box = pred[..., self.nc:]

            # 简化：与targets匹配部分应为IoU匹配/Assign过程
            # 此处暂用虚拟值，仅为展示结构
            target_cls = torch.zeros_like(pred_cls)
            target_box = torch.zeros_like(pred_box)

            cls_loss += self.bce_cls(pred_cls, target_cls)
            box_loss += self.iou_loss(pred_box, target_box).mean()

        total_loss = self.cls_weight * cls_loss + self.box_weight * box_loss
        return total_loss, torch.cat([box_loss[None], cls_loss[None], total_loss[None]])

    def _ciou_loss(self, pred_boxes, target_boxes, eps=1e-7):
        # pred_boxes: [B, N, 4] -> x, y, w, h
        # simplified CIoU
        px, py, pw, ph = pred_boxes.unbind(-1)
        gx, gy, gw, gh = target_boxes.unbind(-1)

        # 转为 corner 格式
        p_x1 = px - pw / 2
        p_y1 = py - ph / 2
        p_x2 = px + pw / 2
        p_y2 = py + ph / 2

        g_x1 = gx - gw / 2
        g_y1 = gy - gh / 2
        g_x2 = gx + gw / 2
        g_y2 = gy + gh / 2

        inter_x1 = torch.max(p_x1, g_x1)
        inter_y1 = torch.max(p_y1, g_y1)
        inter_x2 = torch.min(p_x2, g_x2)
        inter_y2 = torch.min(p_y2, g_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        area_p = pw * ph
        area_g = gw * gh
        union = area_p + area_g - inter_area + eps
        iou = inter_area / union

        return 1.0 - iou  # IoU 损失
