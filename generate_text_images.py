import os
from collections import defaultdict
from collections import namedtuple
from uuid import uuid1
import numpy as np
from random import choice, sample
from PIL import Image, ImageDraw, ImageFont

from multiprocessing import Pool


font_data_dir = '/Users/snowholy/Desktop/还呗设计师常用字体（45种）/'

single_font_data_dir = 'data/train/'

mac_os_file = '.DS_Store'

font_id_txt = 'font_id.txt'

FontBgCombo = namedtuple("FontBgCombo", "fonts text")


def get_all_font():
    font_files = os.listdir(font_data_dir)
    return [font_data_dir + x for x in font_files if x != mac_os_file]



def get_font_id_map():

    fonts = sorted(get_all_font())
    return {x: fonts[x] for x in range(len(fonts))}


def get_text_id_map():
    texts = read_text()
    return {x: texts[x] for x in range(len(texts))}


def mk_single_data():
    fonts_map = get_font_id_map()
    texts_map = get_text_id_map()

    with open(font_id_txt, 'a') as f:
        for item in fonts_map.items():
            f.write("{} {}\n".format(item[0], item[1]))

    res = get_font_combos(fonts_map, texts_map, 1, 15)
    print(len(res))
    return res


def get_font_combos(fonts_dict, texts_dict, k, num_per_font=2):
    font_set = set(fonts_dict.keys())
    text_set = set(texts_dict.keys())

    fonts_combo_nums = defaultdict(int)
    res = []
    while len(font_set) > k:
        font_ids = sample(font_set, k)
        text_id = sample(text_set, 1)[0]

        fonts = [[x, fonts_dict[x]] for x in font_ids]
        ttext = [text_id, texts_dict[text_id]]
        res.append(FontBgCombo(fonts, ttext))
        for font_id in font_ids:
            fonts_combo_nums[font_id] += 1
            if fonts_combo_nums[font_id] >= num_per_font:
                font_set.remove(font_id)

    return res


def read_text():
    file_name = u'data/texts.txt'
    lines =[]
    with open(file_name, 'r') as f:  # , encoding='utf-8'
        # texts = f.read().split(' ')
        for line in f.readlines():
            lines.append(line)

        return lines


def draw_pic(font_text_combo):
    fonts = font_text_combo.fonts
    fonts_id = [x[0] for x in fonts]
    ttext = font_text_combo.text
    id = str(uuid1())

    image = Image.new("RGB", (5000, 5000), (0, 0, 0))
    position = (20, 20)
    for font in fonts:
        # 字体设置
        font_size = set(range(36, 120))
        font_size = sample(font_size, 1)[0]
        draw = ImageDraw.Draw(image)
        my_font = ImageFont.truetype(font[1], size=font_size)
        draw.multiline_text(position, ttext[1], font=my_font, fill=(10, 10, 255))

        im_slice = np.asarray(image)[:, :, 0]
        y, x = np.where(im_slice != 0)
        x_max, x_min, y_max, y_min = np.max(x), np.min(x), np.max(y), np.min(y)
        frame = 10
        box = (x_min - frame, y_min - frame, x_max + frame, y_max + frame)
        text_image=image.crop(box)
        out_dir = os.path.join(single_font_data_dir, font[1].split('/')[-1].split('.')[0])
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        text_image.save((out_dir + "/{}_{}.jpg").format(id, fonts_id))


if __name__ == '__main__':
    read_text()
    res = mk_single_data()
    pool_num = 10
    p = Pool(pool_num)
    draw_pic(res[0])
    p.map(draw_pic, res)
