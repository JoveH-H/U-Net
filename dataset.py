import io
import numpy as np
from PIL import Image as PilImage
from paddle.io import Dataset
from paddle.vision.transforms import transforms as T


class PetDataset(Dataset):
    """
    数据集定义
    """

    def __init__(self, mode='train', image_size=(160, 160), resources_path="./resources/Oxford-IIIT Pet"):
        """
        构造函数
        """
        self.image_size = image_size
        self.mode = mode.lower()

        assert self.mode in ['train', 'test'], "mode should be 'train' or 'test', but got {}".format(self.mode)

        self.origin_images = []
        self.label_images = []

        with open(resources_path + '/{}.txt'.format(self.mode), 'r') as f:
            for line in f.readlines():
                image, label = line.strip().split('\t')
                self.origin_images.append(image)
                self.label_images.append(label)

    def _load_img(self, path, color_mode='rgb', transforms=[]):
        """
        统一的图像处理接口封装，用于规整图像大小和通道
        """
        with open(path, 'rb') as f:
            img = PilImage.open(io.BytesIO(f.read()))
            if color_mode == 'grayscale':
                # if image is not already an 8-bit, 16-bit or 32-bit grayscale image
                # convert it to an 8-bit grayscale image.
                if img.mode not in ('L', 'I;16', 'I'):
                    img = img.convert('L')
            elif color_mode == 'rgba':
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
            elif color_mode == 'rgb':
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            else:
                raise ValueError('color_mode must be "grayscale", "rgb", or "rgba"')
            
        if self.mode == 'train' and color_mode == 'rgb':
            train_transforms = [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)]
            return T.Compose(train_transforms + [T.Resize(self.image_size)] + transforms)(img)
        else:
            return T.Compose([T.Resize(self.image_size)] + transforms)(img)

    def __getitem__(self, idx):
        """
        返回 image, label
        """
        origin_image = self._load_img(self.origin_images[idx], color_mode='rgb', transforms=[T.Transpose(), T.Normalize(mean=127.5, std=127.5)])  # 加载原始图像
        label_image = self._load_img(self.label_images[idx], color_mode='grayscale', transforms=[T.Grayscale()])  # 加载Label图像

        # 返回image, label
        origin_image = np.array(origin_image, dtype='float32')
        label_image = np.array(label_image, dtype='int64')

        return origin_image, label_image

    def __len__(self):
        """
        返回数据集总数
        """
        return len(self.origin_images)