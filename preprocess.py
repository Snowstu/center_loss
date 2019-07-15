# 2019年6月19日
# snow:制作标签

import shutil
import os
from PIL import Image


def move_file():
    path = ""
    path_dir = ""
    paths_ = os.listdir(path)
    paths_.remove(".DS_Store")

    for p in paths_:
        image_dir = os.path.join(path, p)
        for img in os.listdir(image_dir):

            if img.startswith(".DS_"):
                continue

            ii = img.split(".")[0]
            pth = os.path.join(path_dir, p)
            if not os.path.exists(pth):
                os.mkdir(pth)
            dst = pth + '/' + "{}.jpg".format(ii)
            shutil.move(os.path.join(image_dir, img), dst)


def generate_data(p):

    if not os.path.exists(os.path.join(p,'train')):
        os.mkdir(os.path.join(p,'train'))
    if not os.path.exists(os.path.join(p,'val')):
        os.mkdir(os.path.join(p,'val'))

    p1 = os.path.join(p, 'train')
    p2 = os.path.join(p, 'val')

    cnt = 0
    for f in os.listdir(p):
        if f.startswith(".DS_"):
            continue
        if not f.endswith('.jpg'):
            continue

        cnt += 1

        if cnt % 4 == 0:
            val_path = os.path.join(p2, f)

            shutil.move(os.path.join(p, f), val_path)
        else:
            train_path = os.path.join(p1, f)
            shutil.move(os.path.join(p, f), train_path)


def generate_labels(p, num):
    # p = os.getcwd()

    p1 = os.path.join(p, 'train')
    p2 = os.path.join(p, 'val')

    f1 = open(os.path.join(p, 'train_label.txt'), 'a')
    f2 = open(os.path.join(p, 'val_label.txt'), 'a')

    for f in os.listdir(p1):
        if f.startswith(".DS_"):
            continue
        labels1 = f.split('_')
        train_label = ''
        for i in range(len(labels1[1:])):
            if i > num:
                continue
            train_label += ' ' + str(labels1[i])

        train_label += '\n'
        train_path = os.path.join(p1, f)
        line1 = train_path + train_label
        f1.write(line1)

    for ff in os.listdir(p2):
        if ff.startswith(".DS_"):
            continue
        labels2 = ff.split('_')
        val_label = ''
        for i in range(len(labels2)):
            if i > num:
                continue
            val_label += ' ' + str(labels2[i])

        val_label += '\n'
        val_path = os.path.join(p2, ff)
        line2 = val_path + val_label
        f2.write(line2)


def clean():
    curDir = 'dataset/val'
    badFilesList = []
    for root, dirs, files in os.walk(curDir):
        # print(files)
        # 检查当前目录中的损坏的图片文件
        for each in files:
            # for each in os.listdir('./'):
            if each.endswith('.png') or each.endswith('.jpg') or each.endswith('.gif') or each.endswith(
                    '.JPG') or each.endswith('.PNG') or each.endswith('.GIF') or each.endswith(
                '.jpeg') or each.endswith(
                '.JPEG'):
                # print(each)

                try:

                    im = Image.open(os.path.join(root, each))
                    # im.show()
                except Exception as e:
                    print('Bad file:', os.path.join(root, each))
                    badFilesList.append(os.path.join(root, each))

        # 删除损坏的文件
    if len(badFilesList) != 0:
        for each in badFilesList:
            try:
                os.remove(each)
            except Exception as e:
                print('Del file: %s failed, %s' % (each, e))

    pass


if __name__ == "__main__":
    # clean()
    generate_data('data')
    # generate_labels('single', 0)
