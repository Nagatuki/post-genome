import os

import cv2
from PIL import Image

import filter


def mkdir(path):
    try:
        os.makedirs(path)

    except FileExistsError as e:
        # ignore
        pass

    except Exception as e:
        raise e


def make_gif():
    file_paths = ["./HQ/frap{:0>2}.tiff".format(i + 1) for i in range(20)]

    # file check
    for file_path in file_paths:
        if os.path.isfile(file_path):
            continue

        print(file_path, "is not a file.")
        return

    imes = []
    for file_path in file_paths:
        im = Image.open(file_path).convert("P")
        imes.append(im)

    imes[0].save(
        "./HQ/all.gif",
        save_all=True,
        append_images=imes[1:],
        optimize=False,
        duration=200,
        loop=0,
    )


def convert_tiff_to_png():
    file_paths = ["./HQ/frap{:0>2}.tiff".format(i + 1) for i in range(20)]
    imgs = [cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) for file_path in file_paths]

    for i, raw_img in enumerate(imgs):
        # raw
        dir_path = "./output/temp/raw/"
        mkdir(dir_path)
        cv2.imwrite(dir_path + "{:0>2}.png".format(i + 1), raw_img)

        # moving average
        img = filter.moving_average(raw_img, ksize=9)
        dir_path = "./output/temp/mov_ave/"
        mkdir(dir_path)
        cv2.imwrite(dir_path + "{:0>2}.png".format(i + 1), img)

        # gaussian
        img = filter.gaussian_filter(raw_img, ksize=9, sigma=1.7)
        dir_path = "./output/temp/gaussian/"
        mkdir(dir_path)
        cv2.imwrite(dir_path + "{:0>2}.png".format(i + 1), img)
