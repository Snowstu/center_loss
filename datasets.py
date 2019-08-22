import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


# IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
#                   '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ]
#
#
# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(dir):
    fonts = sorted(os.listdir(dir))
    fonts.remove(".DS_Store")

    classes = [d for d in fonts]
    class_to_idx = {}
    for i in range(len(classes)):
        class_to_idx[i] = classes[i]
    return classes, class_to_idx


def make_dataset(root, mode):
    images = []

    for line in os.listdir(os.path.join(root, mode)):

        image_name = os.path.join(root, mode) + '/' + line
        if not image_name.endswith('.jpg'):
            continue
        item = line.split('_')
        label = json.loads(item[-2])
        images.append((image_name, label[0]))

    return images


class ImageFolder(Dataset):
    def __init__(self, root, size, mode):
        imgs = make_dataset(root, mode)
        self.imgs = imgs
        self.input_size = size
        if len(imgs) == 0:
            raise (RuntimeError('Found 0 images in folders of :' + root))

    def __getitem__(self, index):
        image_path, labels = self.imgs[index]
        em_label = np.zeros(109)
        em_label[labels] = 1  # 单标签
        # em_label = labels

        img = Image.open(image_path)
        img = img.convert("RGB")
        max_side_dest_length = self.input_size
        max_side_length = max(img.size)
        ratio = max_side_dest_length / max_side_length
        new_size = [int(ratio * x) for x in img.size[::-1]]
        new_size=[299,299]

        h = new_size[0]
        w = new_size[1]
        pad = [0, 0]

        if h > w:
            pad = [(h - w), 0, 0, 0]
        else:
            pad = [0, (w - h), 0, 0]

        img_loader = transforms.Compose([
            transforms.Resize(new_size),
            transforms.Pad(padding=tuple(pad)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        ])
        image = img_loader(img)
        return image, torch.LongTensor(em_label)  # LongTensor

    def __len__(self):
        return len(self.imgs)
