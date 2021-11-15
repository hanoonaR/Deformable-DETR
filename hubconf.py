import torch
from models.backbone import Backbone, Joiner
from models.deformable_detr import DeformableDETR, PostProcess
from models.position_encoding import PositionEmbeddingSine
from models.deformable_transformer import DeformableTransformer

dependencies = ["torch", "torchvision"]


def _make_deformable_detr(backbone_name: str, dilation=False, num_classes=91):
    hidden_dim = 256
    backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=True, dilation=dilation)
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    transformer = DeformableTransformer(d_model=hidden_dim, return_intermediate_dec=True)
    deformable_detr = DeformableDETR(backbone_with_pos_enc, transformer, num_classes=num_classes, num_queries=300,
                                     num_feature_levels=4)
    return deformable_detr


def deformable_detr_resnet101(pretrained=False, return_postprocessor=False, checkpoints_path=""):
    """
    Deformable DETR R101 with 6 encoder and 6 decoder layers.
    """
    num_classes = 91
    model = _make_deformable_detr("resnet101", dilation=False, num_classes=num_classes)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    elif checkpoints_path:
        checkpoint = torch.load(checkpoints_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model
