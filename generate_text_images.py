import os
import sys
from collections import defaultdict
from collections import namedtuple
from uuid import uuid1
import random
from random import choice, sample
from PIL import Image, ImageDraw, ImageFont

from multiprocessing import Pool

back_ground_dir = 'bg/'

font_data_dir = 'font_dfp/'

single_font_data_dir = '/data'

mac_os_file = '.DS_Store'

font_id_txt = 'font_id_dfp.txt'
bg_id_txt = 'text_id_dfp.txt'

FontBgCombo = namedtuple("FontBgCombo", "fonts text")



def get_all_font():
    font_files = os.listdir(font_data_dir)
    return [font_data_dir + x for x in font_files if x != mac_os_file]


def get_all_bg():
    bg_files = os.listdir(back_ground_dir)
    return [back_ground_dir + x for x in bg_files if x != mac_os_file]


def get_font_id_map():
    fonts = get_all_font()
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

    res = get_bg_font_combos(fonts_map, texts_map, 1, 10)
    print(len(res))
    return res


def get_bg_font_combos(fonts_dict, texts_dict, k, num_per_font=100):
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
    file_name = u'/Users/snowholy/center_loss/data/texts.txt'
    with open(file_name, 'r') as f:  # , encoding='utf-8'
        texts = f.read()

    # start = random.randint(0, 1000)
    # num = random.randint(6, 10)
    # s = texts[start]
    # for j in range(num):
    #     s += texts[start + j]
    # if (1 + j) % 9== 0:
    #     s += '\n'

    # texts = "身 份 证 提 现"
    return texts.split(" ")



def draw_pic(font_text_combo):
    fonts = font_text_combo.fonts
    fonts_id = [x[0] for x in fonts]
    ttext = font_text_combo.text
    text_id = ttext[0]
    id = str(uuid1())

    text_image = Image.new("RGB", (1000, 1000), (255, 255, 255))

    for font in fonts:
        # 字体设置
        font_size = set(range(16, 80))
        font_size = sample(font_size, 1)[0]
        draw = ImageDraw.Draw(text_image)
        my_font = ImageFont.truetype(font[1], size=font_size)

        text_size = draw.textsize(ttext[1], font=my_font)
        w, h = text_image.width, text_image.height
        x = random.randint(0, w - text_size[0])
        y = random.randint(0, h - text_size[1])

        position = (x, y)
        draw = ImageDraw.Draw(text_image)
        draw.multiline_text(position, ttext[1], font=my_font, fill=(0, 0, 0))

        text_image = text_image.crop((x, y, x + text_size[0], y + text_size[1]))

    text_image.save((single_font_data_dir + "{}_{}_{}.jpg").format(id, fonts_id, text_id))  # [x+109 for x in fonts_id]


if __name__ == '__main__':

    res = mk_single_data()
    pool_num = 10
    p = Pool(pool_num)
    draw_pic(res[0])
    p.map(draw_pic, res)
