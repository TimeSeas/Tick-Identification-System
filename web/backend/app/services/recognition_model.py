"""Shared, local-only tick presence and species model for rebuild experiments."""
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F_torch
from torchvision.models import resnet50
from torchvision.transforms import functional as F

MEAN = (.485, .456, .406)
STD = (.229, .224, .225)
PREPROCESSING = 'exif_then_xyxy_context015_letterbox_imagenet_v1'


def prepare_image(image, bbox=None, context=.15, normalize_exif=True):
    image = (ImageOps.exif_transpose(image) if normalize_exif else image).convert('RGB')
    if bbox is not None:
        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Invalid region box')
        dx, dy = (x2-x1)*context, (y2-y1)*context
        box = (max(0, int(x1-dx)), max(0, int(y1-dy)),
               min(image.width, int(x2+dx+.999)), min(image.height, int(y2+dy+.999)))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError('Region outside image')
        image = image.crop(box)
    return image


def letterbox_image(image, size=448):
    ratio = size/max(image.size)
    resized = image.resize((max(1, round(image.width*ratio)), max(1, round(image.height*ratio))),
                           Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (size, size), tuple(round(v*255) for v in MEAN))
    canvas.paste(resized, ((size-resized.width)//2, (size-resized.height)//2))
    return canvas


def letterbox_tensor(image, size=448):
    return F.normalize(F.to_tensor(letterbox_image(image, size)), MEAN, STD)


class RecognitionModel(nn.Module):
    def __init__(self, num_species):
        super().__init__()
        self.encoder = resnet50(weights=None)
        self.encoder.fc = nn.Identity()
        self.tick_head = nn.Linear(2048, 2)
        self.species_head = nn.Linear(2048, num_species)

    def forward(self, images):
        features = self.encoder(images)
        return dict(tick_logits=self.tick_head(features), species_logits=self.species_head(features))


def recognition_loss(output, tick_labels, species_labels):
    tick_loss = F_torch.cross_entropy(output['tick_logits'], tick_labels)
    known = species_labels >= 0
    species_loss = (F_torch.cross_entropy(output['species_logits'][known], species_labels[known],
                                        label_smoothing=.05) if known.any()
                    else output['species_logits'].sum()*0)
    return tick_loss+species_loss
