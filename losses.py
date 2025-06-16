import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ops.tree_filter.modules.tree_filter import MinimumSpanningTree, TreeFilter2D
## new
class TreeEnergyLoss(nn.Module):
    def __init__(self, configer=None):
        super(TreeEnergyLoss, self).__init__()
        self.configer = configer
        if self.configer is None:
            print("self.configer is None")

        self.weight = 1.0
        self.mst_layers = MinimumSpanningTree(TreeFilter2D.norm2_distance)
        self.tree_filter_layers = TreeFilter2D(groups=1, sigma=0.002)

    def forward(self, preds, low_feats, high_feats, unlabeled_ROIs):
        # scale low_feats via high_feats
        with torch.no_grad():
            batch, _, h, w = preds.size()
            low_feats = F.interpolate(low_feats, size=(h, w), mode='bilinear', align_corners=False)
            unlabeled_ROIs = F.interpolate(unlabeled_ROIs.unsqueeze(1).float(), size=(h, w), mode='nearest')
            N = unlabeled_ROIs.sum()

        prob = torch.softmax(preds, dim=1)

        # low-level MST
        tree = self.mst_layers(low_feats)
        AS = self.tree_filter_layers(feature_in=prob, embed_in=low_feats, tree=tree)  # [b, n, h, w]

        # high-level MST
        if high_feats is not None:
            tree = self.mst_layers(high_feats)
            AS = self.tree_filter_layers(feature_in=AS, embed_in=high_feats, tree=tree, low_tree=False)  # [b, n, h, w]

        tree_loss = (unlabeled_ROIs * torch.abs(prob - AS)).sum()
        if N > 0:
            tree_loss /= N

        return self.weight * tree_loss
    
def _unfold_wo_center(x, kernel_size, dilation, with_center=False):
    """
    x: [bsz, c, h, w]
    kernel_size: k
    dilation: d
    return: [bsz, c, k**2-1, h, w]
    """

    assert x.ndim == 4, x.shape
    assert kernel_size % 2 == 1, kernel_size

    padding = (kernel_size + (dilation - 1) * (kernel_size - 1)) // 2
    unfolded_x = F.unfold(x, kernel_size=kernel_size, dilation=dilation, padding=padding)

    n, c, h, w = x.shape
    unfolded_x = unfolded_x.reshape(n, c, -1, h, w)

    if with_center:
        return unfolded_x

    # remove the center pixel
    size = kernel_size**2
    unfolded_x = torch.cat((unfolded_x[:, :, :size // 2], unfolded_x[:, :, size // 2 + 1:]), 2)
    return unfolded_x

@torch.no_grad()
def _normalized_rgb_to_lab(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    images: [bsz, 3, h, w]
    """
    assert images.ndim == 4, images.shape
    assert images.shape[1] == 3, images.shape

    device = images.device
    mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
    images = (images * std + mean).clip(min=0, max=1)
    rgb = images

    mask = (rgb > .04045).float()
    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)
    xyz_const = torch.as_tensor([
        .412453,.357580,.180423,
        .212671,.715160,.072169,
        .019334,.119193,.950227], device=device).view(3, 3)
    xyz = torch.einsum('mc,bchw->bmhw', xyz_const, rgb)

    sc = torch.as_tensor([0.95047, 1., 1.08883], device=device).view(1, 3, 1, 1)
    xyz_scale = xyz / sc
    mask = (xyz_scale > .008856).float()
    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)
    lab_const = torch.as_tensor([
        0., 116., 0.,
        500., -500., 0.,
        0., 200., -200.], device=device).view(3, 3)
    lab = torch.einsum('mc,bchw->bmhw', lab_const, xyz_int)
    lab[:, 0] -= 16
    return lab.float()

def get_penalty(predict, cls_label):
    # cls_label: (n, c)
    # predict: (n, c, h, w)
    n, c, h, w = predict.size()
    predict = torch.softmax(predict, dim=1)

    # if a patch does not contain label c,
    # then none of the pixels in this patch can be assigned to label c
    loss0 = - (1 - cls_label.view(n, c, 1, 1)) * torch.log(1 - predict + 1e-6)
    loss0 = torch.mean(torch.sum(loss0, dim=1))

    # if a patch has only one type, then the whole patch should be assigned to this type
    sum = (torch.sum(cls_label, dim=-1, keepdim=True) == 1)
    loss1 = - (sum * cls_label).view(n, c, 1, 1) * torch.log(predict + 1e-6)
    loss1 = torch.mean(torch.sum(loss1, dim=1))
    return loss0 + loss1

def color_prior_loss(data, images, masks=None, dilation=2, avg_factor=None):
    """
    data:   [bsz, classes, h, w] or [bsz, h, w]
    images: [bsz, 3, h, w]
    masks:  [bsz, h, w], (opt.), valid regions
    """
    if data.ndim == 4:
        log_prob = F.log_softmax(data, 1)
    elif data.ndim == 3:
        log_prob = torch.cat([F.logsigmoid(-data[:, None]), F.logsigmoid(data[:, None])], 1)
    else:
        raise ValueError(data.shape)

    B, C, H, W = data.shape
    assert images.shape == (B, 3, H, W), (images.shape, data.shape)
    if masks is not None:
        assert masks.shape == (B, H, W), (masks.shape, data.shape)

    images = _normalized_rgb_to_lab(images)

    kernel_size_list = [3, 5]
    weights = [0.35, 0.65]
    loss = []

    for kernel_size, weight in zip(kernel_size_list, weights):

        log_prob_unfold = _unfold_wo_center(log_prob, kernel_size, dilation) # [bsz, classes, k**2-1, h, w]
        log_same_prob = log_prob[:, :, None] + log_prob_unfold
        max_ = log_same_prob.max(1, keepdim=True)[0]
        log_same_prob = (log_same_prob - max_).exp().sum(1).log() + max_.squeeze(1) # [bsz, k**2-1, h, w]

        images_unfold = _unfold_wo_center(images, kernel_size, dilation)
        images_diff = images[:, :, None] - images_unfold
        images_sim = (-torch.norm(images_diff, dim=1) * 0.5).exp() # [bsz, k**2-1, h, w]

        loss_weight = (images_sim >= 0.3).float()

        if masks is not None:
            loss_weight = loss_weight * masks[:, None]

        loss_color = -(log_same_prob * loss_weight).sum((1, 2, 3)) / loss_weight.sum((1, 2, 3)).clip(min=1)
        loss_color = loss_color.sum() / (len(loss_color) if avg_factor is None else avg_factor)
        loss.append(weight * loss_color)

    return sum(loss)

def rgd_semantic_loss(logits, mask_targets, image):

    "long-range rbg-color loss"

    mst = MinimumSpanningTree(TreeFilter2D.norm2_distance)
    tree_filter = TreeFilter2D()

    # labeled_region = torch.sum(mask_targets, dim=1).unsqueeze(dim=1)
    # unlabled_region = 1.0 - labeled_region
    # N = unlabled_region.sum()
    unlabled_region = (mask_targets==255).unsqueeze(1)

    prob = torch.softmax(logits, dim=1)
    tree = mst(image)
    rgb_affinity = tree_filter(feature_in=prob, embed_in=image, tree=tree)
    # 预测的语义图概率prob和基于图像颜色的解构相似性之间的差异
    loss_tree_color = (unlabled_region*(torch.abs(prob - rgb_affinity))).sum() / unlabled_region.sum()

    return loss_tree_color