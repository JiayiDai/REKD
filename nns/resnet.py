import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RESNET(nn.Module):
    def __init__(self, args, encoding=None, if_t=False):
        super(RESNET, self).__init__()
        self.encoding = encoding
        if if_t:
            model_form = args.model_form_t
            if model_form not in ["resnet18", "resnet50", "resnet101"]:
                raise NotImplementedError("Model form {} not yet supported for teacher!".format(model_form))
        else:
            model_form = args.model_form
            if model_form not in ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "efficientnet_b7"]:
                 raise NotImplementedError("Model form {} not yet supported!".format(model_form))
        if model_form == "resnet18":
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_form == "resnet34":
            self.resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_form == "resnet50":
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        elif model_form == "resnet101":
            self.resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        elif model_form == "resnet152":
            self.resnet = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        elif model_form == "efficientnet_b7":
            self.resnet = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        else:
            raise NotImplementedError("Model form {} not yet supported!".format(model_form))

        if self.encoding:
            original_weights = self.resnet.conv1.weight.data
            new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            new_conv1.weight.data = weight_adaption(original_weights)
            self.resnet.conv1 = new_conv1
            self.resnet.maxpool = nn.Identity()
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, args.num_class)
            
        else:
            self.gen = smp.UnetPlusPlus(#PlusPlus
                encoder_name=model_form,
                encoder_weights="imagenet",#
                in_channels=3,
                classes=2
            )
            """
            original_weights = self.gen.encoder.conv1.weight.data
            new_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            new_conv1.weight.data = weight_adaption(original_weights)
            self.gen.encoder.conv1 = new_conv1
            self.gen.encoder.maxpool = nn.Identity()
            """

    def forward(self, x):
        if self.encoding:
            return self.resnet(x)
        else:
            return self.gen(x)

def weight_adaption(original_weights, method="cropped"):
    with torch.no_grad():
        if method == "cropped":
            adapted_weights = original_weights[:, :, 2:5, 2:5]
        elif method == "interpolated":
            adapted_weights = F.interpolate(
                original_weights, 
                size=(3, 3), 
                mode='bilinear', 
                align_corners=False
            )
        elif method == "avg_pooled":
            adapted_weights = F.adaptive_avg_pool2d(original_weights, (3, 3))
        else:
            raise NotImplementedError("Weight adaption method {} not yet supported!".format(method))
        return adapted_weights