import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss, KLDivLoss
from einops.layers.torch import Rearrange
from torch.autograd import Variable
import math
try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse

class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, smooth=1, p=2, ignore_index=0, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.ignore_index = ignore_index
        self.reduction=reduction

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
       
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()]#마찬가지로 추려냄

        return vprobas.contiguous().view(-1), vlabels.contiguous().view(-1)#추려낸 뒤에 펴서 리턴
    
    def forward(self, predicts, targets):
        loss_total=[]
        for predict, target in zip(predicts, targets):# 배치의 샘플 단위로 손실 값 측정
            predict = predict.unsqueeze(0)#(1, C, H, W)
            target = target.unsqueeze(0)#(1, C, H, W)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# #(1, C, H, W) -> (K*C)
            predict = predict.unsqueeze(0)#(1, K*C)
            target = target.unsqueeze(0)#(1, K*C)

            num = 2 * (torch.sum(predict * target, dim=1) + self.smooth)
            den = torch.sum(predict ** self.p, dim=1) + torch.sum(target ** self.p, dim=1) + self.smooth

            loss = 1 - num / den
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)
        
        return loss

class CategoricalCrossEntropyLoss(nn.Module):
    """Categorical Cross Entropy loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as CrossEntropyLoss
    """
    def __init__(self, weight=None, ignore_index=0, reduction='mean', **kwargs):
        super(CategoricalCrossEntropyLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction=reduction
        self.CEloss = CrossEntropyLoss(reduction=self.reduction,**self.kwargs)

    def forward(self, predicts, targets):
        if self.ignore_index==0:
            targets = targets[:,1:,:]# one-hot
            predicts = predicts[:,1:,:]# predicted prob.

        assert predicts.shape == targets.shape, 'predict & target shape do not match'
        # predict = F.softmax(predict, dim=1)  # prob

        term_true = - torch.log(predicts)
        term_false = - torch.log(1-predicts)
        loss = torch.sum(term_true * targets + term_false * (1-targets), dim=1) #torch.Size([8, 256, 512])

        if self.reduction == "mean":# torch.Size([]) # loss: 4.507612  [    0/ 2975]
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass       
        
        # loss = self.CEloss(predict,torch.argmax(target, dim=1)) # torch.Size([]) # loss: 3.569603  [    0/ 2975]
        return loss

class FocalLoss(nn.Module):
    """Focal loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str ='mean'):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
       
        if probas.dim() == 4:
            # 2D segmentation
            B, C, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 1).contiguous().view(-1, C) # (B=1)*H*W, C

        if labels.dim() == 4:# B,C,H,W -> B,H,W
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,H,W -> B*H*W

        if labels.dim() == 4:# B,C,H,W -> B*H*W, C
            # assumes output of a sigmoid layer
            B, C, H, W = labels.size()
            labels = labels.view(B, C, H, W).permute(0, 2, 3, 1).contiguous().view(-1, C)# (B=1)*H*W, C
        
        if ignore is None:
            return probas, labels

        valid = (labels_arg != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()] #마찬가지로 추려냄

        return vprobas, vlabels
    
    def forward(self, predicts: Tensor, targets: Tensor):     
        loss_total=[]
        for predict, target in zip(predicts, targets):# 배치의 샘플 단위로 손실 값 측정
            predict = predict.unsqueeze(0)#(1, C, H, W)
            target = target.unsqueeze(0)#(1, C, H, W)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# #(1, C, H, W) -> (K,C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) # 틀리면 손실 커짐, 맞을수록 작아짐

            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)

class Focal_3D_loss(nn.Module):
    def __init__(self, alpha:float = 0.25, gamma:float = 2, eps = 1e-8, ignore_index=0, reduction:str = 'mean'):
        super(Focal_3D_loss, self).__init__()
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reduction = reduction
    
    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)

        if labels.dim() == 3:# B,C,N -> B,N
            labels_arg = torch.argmax(labels, dim=1)
            labels_arg = labels_arg.view(-1)# B,N -> B*N

        if labels.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = labels.size()
            labels = labels.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)

        if ignore is None:
            return probas, labels
        
        valid = (labels_arg != ignore)
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] 
        vlabels = labels[valid.nonzero(as_tuple=False).squeeze()]

        return vprobas, vlabels
    
    def forward(self, predicts:Tensor, targets:Tensor):
        loss_total=[]
        for predict, target in zip(predicts, targets):
            predict = predict.unsqueeze(0)
            target = target.unsqueeze(0)
            
            predict, target = self.flatten_probas(predict, target, ignore=0)# (1*K, C) | (1*K, C)

            term_true =  - self.alpha * ((1 - predict) ** self.gamma) * torch.log(predict+self.eps)
            term_false = - (1-self.alpha) * (predict**self.gamma) * torch.log(1-predict+self.eps) 
            
            loss = torch.sum(term_true * target + term_false * (1-target), dim=-1)# (1*K) 
            
            loss_total.append(loss)
 
        if self.reduction == "mean":
            return torch.mean(torch.cat(loss_total))
        elif self.reduction == "sum":
            return torch.sum(torch.cat(loss_total))
        elif self.reduction == "none":
            return torch.cat(loss_total)

class FocalLosswithDiceRegularizer(nn.Module):
    """Focal loss with Dice loss as regularizer, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryFocalLoss
    Return:
        same as BinaryFocalLoss
    """
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2,  ignore_index=0, reduction:str ='sum'):
        super(FocalLosswithDiceRegularizer, self).__init__()
        self.ignore_index = ignore_index        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.smooth = smooth
        self.p = p
        self.focal_loss = FocalLoss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_index, reduction=reduction)
        self.dice_regularizer = DiceLoss(smooth=self.smooth, p=self.p, ignore_index = self.ignore_index, reduction=reduction)

    def forward(self, predict: Tensor, target: Tensor):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        # predict = F.softmax(predict, dim=1) + self.eps # prob
        f_loss = self.focal_loss(predict, target)
        # d_regularization = self.dice_regularizer(predict*target, target)
        # return f_loss + (8 * d_regularization)
        d_loss = self.dice_regularizer(predict, target)
        return f_loss + d_loss

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, classes, ignore_index, smoothing=0.1, dim=-1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        if self.ignore_index==0:
            target = target[:,1:,:]# one-hot
            pred = pred[:,1:,:]# predicted prob.

        pred = pred.log_softmax(dim=self.dim)
        target = torch.argmax(target, dim=1)
        # true_dist = pred.data.clone()
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.cls - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)

        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

class Lovasz_loss(nn.Module):
    def __init__(self, reduction:str='mean', ignore_index:int = 0):
        super(Lovasz_loss, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_index
    
    def mean(self, l, ignore_nan=False, empty=0):
        """
        nanmean compatible with generators.
        """    
        def isnan(x):
            return x != x

        l = iter(l)
        if ignore_nan:
            l = ifilterfalse(isnan, l)
        try:
            n = 1
            acc = next(l)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(l, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def lovasz_grad(self, gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1] 
        return jaccard   


    def lovasz_softmax_flat(self, probas, labels, classes='present'):
        """
        Multi-class Lovasz-Softmax loss
        probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
        labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.
        C = probas.size(1)#클래스 수
        losses = []
        class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
        for c in class_to_sum:
            fg = (labels == c).float()  # foreground for class c
            if (classes == 'present' and fg.sum() == 0):
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError('Sigmoid output possible only with 1 class')
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (Variable(fg) - class_pred).abs()# target - pred
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, Variable(self.lovasz_grad(fg_sorted)))) # 정렬된 오류와 gradient of the Lovasz extension사이의 내적을 통해 최종 loss전달
        return self.mean(losses)

    def flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch
        """
        if probas.dim() == 3:# B,C,N -> B*N, C
            # assumes output of a sigmoid layer
            B, C, N = probas.size()
            probas = probas.view(B, C, 1, N).permute(0, 2, 3, 1).contiguous().view(-1, C)
        
        elif probas.dim() == 5:
            # 3D segmentation
            B, C, L, H, W = probas.size()
            probas = probas.contiguous().permute(0, 2, 3, 4, 1).contiguous().view(-1, C)
        # B, C, H, W = probas.size()
        # probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        if labels.dim() == 3:# B,C,N -> B,N
            labels = torch.argmax(labels, dim=1)
        labels = labels.view(-1)# B,N -> B*N
        if ignore is None:
            return probas, labels
        valid = (labels != ignore)#label값이 ignore 아닌 픽셀들만 골라서
        vprobas = probas[valid.nonzero(as_tuple=False).squeeze()] #추려냄
        vlabels = labels[valid]#마찬가지로 추려냄

        return vprobas, vlabels
    
    def lovasz_softmax(self, probas, labels, classes='present', per_image=True, ignore=0):
        """
        Multi-class Lovasz-Softmax loss
        probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
                Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
        labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
        classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        per_image: compute the loss per image instead of per batch
        ignore: void class labels
        """

        if per_image:# mean reduction
            loss = self.mean(self.lovasz_softmax_flat(*self.flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                        for prob, lab in zip(probas, labels))
        else:# sum reduction
            loss = self.lovasz_softmax_flat(*self.flatten_probas(probas, labels, ignore), classes=classes)
        return loss

    def forward(self, uv_out, uv_label):
        lovasz_loss = self.lovasz_softmax(uv_out, uv_label, ignore=self.ignore_idx)
        return lovasz_loss

class FocalLosswithLovaszRegularizer(nn.Module):
    def __init__(self,  alpha:float = 0.75, gamma:float = 2, eps = 1e-8, smooth=1, p=2, reduction:str = 'mean', ignore_idx:int = 0):
        super(FocalLosswithLovaszRegularizer, self).__init__()
        self.reduction = reduction
        self.ignore_idx = ignore_idx
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.Focal_loss = Focal_3D_loss(alpha = self.alpha, gamma = self.gamma, eps = self.eps, ignore_index = self.ignore_idx, reduction=reduction)
        self.Lovasze_loss = Lovasz_loss(reduction=self.reduction, ignore_index=self.ignore_idx)

    def forward(self,pred:Tensor, label:Tensor):
        assert pred.shape == label.shape, 'predict & target shape do not match'
        pred = F.softmax(pred, dim=1)
        f_loss = self.Focal_loss(pred, label)
        # lovasz_regularization = self.Lovasze_loss(pred*label, label)
        # return f_loss + (8 * lovasz_regularization)
        lovasz_loss = self.Lovasze_loss(pred, label)
        return f_loss + lovasz_loss

class total_loss(nn.Module):
    def __init__(self, reduction:str ='sum', ignore_index:int = 0):
        super(total_loss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.loss2D = FocalLosswithDiceRegularizer(reduction=self.reduction, ignore_index = self.ignore_index)
        self.loss_3D = FocalLosswithLovaszRegularizer(reduction=self.reduction, ignore_idx=self.ignore_index)
        # self.KLD_reg_3D = KLDivLoss(reduction='mean')
        self.transform = Rearrange('b e h w  -> b e (h w)')
        self.alpha = 1
        self.klDiv = nn.KLDivLoss(reduction="none")
        
    def mae_2D3D(self, pred_segment, pred_uv):
        pred_segment = F.softmax(pred_segment, dim=1)
        pred_segment = self.transform(pred_segment[:,1:,:,:])

        pred_uv = F.softmax(pred_uv, dim=1)
        pred_uv = pred_uv[:,1:,:]

        se = torch.abs(pred_segment-pred_uv)
        se_sum = se.sum(dim=1)
        scale_mean = se_sum.mean(dim=1)
        return scale_mean.mean()
    
    def kl_div_3D(self, pred_segment, pred_uv):# target, pred
        pred_segment = F.softmax(self.transform(pred_segment), dim=1)
        # pred_segment = pred_segment[:,1:,:,:]

        pred_uv = F.softmax(pred_uv, dim=1)
        # pred_uv = pred_uv[:,1:,:]#B, C, H*W
        losses = (pred_segment * (pred_segment.log() - pred_uv.log())).sum(dim=1)

        return torch.mean(losses)
    
    
    def PerceptionAwareLoss(self, uv_pred, img_pred): # pcd_entropy, img_entropy, pcd_pred, pcd_pred_log, img_pred, img_pred_log
        _,C,_ = uv_pred.shape
        uv_pred = F.softmax(uv_pred, dim=1)
        img_pred = F.softmax(self.transform(img_pred), dim=1)

        # compute uv entropy. entropy는 낮을 수록 정확하고 높을수록 부정확
        uv_pred_log = torch.log(uv_pred.clamp(min=1e-8))
        pcd_entropy = - (uv_pred * uv_pred_log).sum(1) / math.log(C) # compute pcd entropy: p * log p # shape : b n

        # compute img entropy
        img_pred_log = torch.log(img_pred.clamp(min=1e-8))
        img_entropy = - (img_pred * img_pred_log).sum(1) / math.log(C) # compute pcd entropy: p * log p # shape : b n

        # compute Perception Aware Loss
        pcd_confidence = 1 - pcd_entropy 
        img_confidence = 1 - img_entropy 
        information_importance = pcd_confidence - img_confidence 
        pcd_guide_mask = pcd_confidence.ge(0.7).float() 
        img_guide_mask = img_confidence.ge(0.7).float() 
        pcd_guide_weight = information_importance.gt(0).float() * information_importance.abs() * pcd_guide_mask 
        img_guide_weight = information_importance.lt(0).float() * information_importance.abs() * img_guide_mask 

        # compute kl loss
        loss_per_pcd = (self.klDiv(uv_pred_log, img_pred) * img_guide_weight.unsqueeze(1)).mean()
        loss_per_img = (self.klDiv(img_pred_log, uv_pred) * pcd_guide_weight.unsqueeze(1)).mean()
        loss_per = loss_per_pcd + loss_per_img # total per loss값

        return loss_per, pcd_guide_weight, img_guide_weight
    
    def forward(self, segment_out, segment_label, uv_out, uv_label):

        # 2D segmentation loss
        loss_2D_result = self.loss2D(segment_out, segment_label)

        # 3D segmentation loss
        loss_3D_result = self.loss_3D(uv_out, uv_label)

        # mae || KLDiv
        # loss_3D_regularizer = self.kl_div_3D(segment_out.detach(), uv_out) * self.alpha 

        # return loss_2D_result + loss_3D_result + loss_3D_regularizer.clamp(min=1e-8)

        # PerceptionAwareLoss 사용할 시
        # loss_per, pcd_guide_weight, img_guide_weight = self.PerceptionAwareLoss(uv_out, segment_out)
        return loss_2D_result + loss_3D_result
