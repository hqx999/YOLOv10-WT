import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedDetectionLoss(nn.Module):
    def __init__(self, num_classes=80, lambda_box=5.0, lambda_obj=1.0, lambda_cls=1.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, preds, targets):
        """
        preds: Tensor, shape [B, num_anchors, H, W, 5+num_classes]
        targets: Tensor, shape [B, max_objs, 5] (每行: [cls, x, y, w, h], 均归一化)
        """
        B, A, H, W, D = preds.shape
        device = preds.device
        total_loss = torch.tensor(0.0, device=device)
        
        # 将 targets 扩展为 [B, A*H*W, 5]，如果原始 targets 的形状为 [B, max_objs, 5]
        padded_targets = []
        for i in range(B):
            target = targets[i]  # shape: [N, 5]
            N = target.shape[0]
            desired = A * H * W  # 例如 3 * 40 * 40 = 4800
            padded = torch.zeros((desired, 5), device=device)
            if N > 0:
                # 将前 N 行填上真实目标
                padded[:N, :] = target
            padded_targets.append(padded)
        padded_targets = torch.stack(padded_targets, dim=0)  # [B, A*H*W, 5]
        
        # 简单示例：对每个 batch，我们只使用第一个有效目标进行损失计算
        for i in range(B):
            pred = preds[i].view(-1, 5 + self.num_classes)  # [A*H*W, 5+num_classes]
            target = padded_targets[i]  # [A*H*W, 5]
            
            # 找出第一个非零目标（只作为示例，真实情况需匹配多个目标）
            valid = (target.sum(dim=1) != 0)
            if valid.sum() == 0:
                continue
            idx = valid.nonzero()[0].item()
            gt = target[idx]  # [5]
            gt_cls = gt[0].long()
            gt_box = gt[1:]
            
            # 取第一个预测作为示例匹配
            pred_sample = pred[0]  # [5+num_classes]
            box_pred = pred_sample[:4]
            obj_pred = pred_sample[4]
            cls_pred = pred_sample[5:]
            
            # 计算对象性损失
            obj_loss = F.binary_cross_entropy_with_logits(obj_pred.unsqueeze(0), 
                                                          torch.ones_like(obj_pred.unsqueeze(0)))
            # 计算分类损失（Focal Loss）
            cls_loss_base = F.binary_cross_entropy_with_logits(cls_pred.unsqueeze(0),
                                F.one_hot(gt_cls, num_classes=self.num_classes).float().unsqueeze(0), reduction='none')
            p_t = torch.sigmoid(cls_pred)
            focal_weight = self.focal_alpha * (1 - p_t) ** self.focal_gamma
            cls_loss = (cls_loss_base * focal_weight).mean()
            
            # 计算框回归损失
            box_loss = F.smooth_l1_loss(box_pred, gt_box.expand_as(box_pred))
            
            total_loss += self.lambda_obj * obj_loss + self.lambda_cls * cls_loss + self.lambda_box * box_loss
        
        return total_loss / B
