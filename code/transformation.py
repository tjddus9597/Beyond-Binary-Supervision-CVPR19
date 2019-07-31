from __future__ import absolute_import

from torchvision.transforms import *
from PIL import Image
import math
import random


class ConvertBGR(object):
    def __init__(self):
        pass

    def __call__(self, img):
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
        return img

class RandTranslation(object):
    def __init__(self, size_input, size_crop, s_ratio = 0.05, t_ratio = 0.03):
        self.size_input = size_input 
        self.size_crop = size_crop
        self.s_ratio = s_ratio
        self.t_ratio = t_ratio

    def __call__(self, img):
        
        # for random jittering
        smag_max = math.ceil(self.s_ratio * self.size_crop)
        tmag_max = math.ceil(self.t_ratio * self.size_crop)

        # center 224x224
        xmin = 33
        ymin = 33
        xmax = xmin + 224
        ymax = ymin + 224

        # random scaling
        smag_rand = math.ceil(random.uniform(-smag_max, smag_max))
        xmin = xmin - smag_rand
        xmax = xmax + smag_rand
        ymin = ymin - smag_rand
        ymax = ymax + smag_rand

        # random translation
        tmag_xmin = max(1 - xmin, -tmag_max)
        tmag_ymin = max(1 - ymin, -tmag_max)
        tmag_xmax = min(self.size_input - xmax, tmag_max)
        tmag_ymax = min(self.size_input - ymax, tmag_max)
        tmag_rand_x = math.ceil(random.uniform(tmag_xmin, tmag_xmax))
        tmag_rand_y = math.ceil(random.uniform(tmag_ymin, tmag_ymax))
        xmin = xmin + tmag_rand_x
        xmax = xmax + tmag_rand_x
        ymin = ymin + tmag_rand_y
        ymax = ymax + tmag_rand_y

        img_crop = img.crop((xmin,ymin,xmax,ymax))
        return img_crop

class Scale(object):
    """Rescales the input PIL.Image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class CenterTranslation(object):
    def __init__(self, size_input):
        self.size_input = size_input

    def __call__(self, img):

        # center 224x224
        xmin = 33
        ymin = 33
        xmax = xmin + 224
        ymax = ymin + 224

        img_crop = img.crop((xmin,ymin,xmax,ymax))
        return img_crop