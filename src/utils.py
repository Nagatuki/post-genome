import os

from PIL import Image


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
