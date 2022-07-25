import cv2
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import filter
from fitting import ModelFitting
from utils import mkdir


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


def calc_fluorescence(imgs, center: tuple[int], radius: float) -> tuple[list, list]:
    fluorescence_list = []

    for img in imgs:
        sum_illuminance = 0
        count_pixel = 0

        h, w = img.shape[0], img.shape[1]
        for x in range(h):
            for y in range(w):
                dx = x - center[0]
                dy = y - center[1]

                if dx ** 2 + dy ** 2 <= radius ** 2:
                    sum_illuminance += img[x, y]
                    count_pixel += 1

        fluorescence_list.append(sum_illuminance / count_pixel)

    fluorescence_ratio_list = []
    base = fluorescence_list[0]
    for e in fluorescence_list:
        fluorescence_ratio_list.append(e / base * 100)

    return fluorescence_list, fluorescence_ratio_list


def find_roi(imgs, heavy=False):
    im_pre, im_post = imgs[1], imgs[2]

    # moving average filter
    ksize = 9
    im_pre_filtered = filter.moving_average(im_pre, ksize=ksize)
    im_post_filtered = filter.moving_average(im_post, ksize=ksize)

    # calc fluorescence difference
    im_diff = im_pre_filtered - im_post_filtered
    mkdir("./output/temp/find_roi")
    cv2.imwrite("./output/temp/find_roi/im_diff.png", im_diff)

    # gaussian fitting
    # center, sigma, fwhm = ModelFitting.gaussian2d_fit(im_diff)
    center, sigma, fwhm = ModelFitting.gaussian2d_fit2(im_diff)

    # # check fitting result
    im_diff_cpy = im_diff.copy()
    box = [[center[1], center[0]], [2 * sigma[1], 2 * sigma[0]], 0.0]
    cv2.ellipse(im_diff_cpy, box, color=255, thickness=1)
    cv2.imwrite("./output/temp/find_roi/found_roi_sigma.png", im_diff_cpy)

    im_diff_cpy = im_diff.copy()
    box = [[center[1], center[0]], [fwhm[1], fwhm[0]], 0.0]
    cv2.ellipse(im_diff_cpy, box, color=255, thickness=1)
    cv2.imwrite("./output/temp/find_roi/found_roi_fwhm.png", im_diff_cpy)

    return center, sigma


def find_roi2(imgs, heavy=False):
    im_pre, im_post = imgs[1], imgs[2]

    # moving average filter
    ksize = 9
    im_pre_filtered = filter.moving_average(im_pre, ksize=ksize)
    im_post_filtered = filter.moving_average(im_post, ksize=ksize)

    # calc fluorescence difference
    im_diff = im_pre_filtered - im_post_filtered

    mkdir("./output/temp/find_roi")
    cv2.imwrite("./output/temp/find_roi/im_diff.png", im_diff)

    # gaussian fitting
    center, radius = ModelFitting.quadratic2d_fit2(im_diff, True)

    # # check fitting result
    im_diff_cpy = im_diff.copy()
    cv2.circle(im_diff_cpy, [center[1], center[0]], radius, color=255, thickness=1)
    cv2.imwrite("./output/temp/find_roi/found_roi_radius.png", im_diff_cpy)

    return center, [radius, radius]


def examine_roi(imgs, center, sigma):
    # find best radius
    sigma_coefs = np.arange(0.2, 3.1, 0.2)
    fluorescence_decrease = []
    fluorescence_recovery = []
    fluorescence_steady = []
    for sigma_coef in sigma_coefs:
        radius = [s * sigma_coef for s in sigma]
        _, fluorescence_ratio_list = calc_fluorescence(imgs, center, radius)

        decrease = 100 - fluorescence_ratio_list[2]
        fluorescence_decrease.append(decrease)

        recovery = fluorescence_ratio_list[-1] - fluorescence_ratio_list[2]
        fluorescence_recovery.append(recovery)

        fluorescence_steady.append(fluorescence_ratio_list[-1])

    # # compare about fluorescence decrease
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sigma_coefs, fluorescence_decrease)
    ax.set_xlabel("sigma coefficients")
    ax.set_ylabel("Fluorescence decrease")

    mkdir("./output/temp/fluorescence/")
    plt.savefig("./output/temp/fluorescence/decrease.png")

    # # compare about fluorescence recovery
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sigma_coefs, fluorescence_recovery)
    ax.set_xlabel("sigma coefficients")
    ax.set_ylabel("Fluorescence recovery")

    mkdir("./output/temp/fluorescence/")
    plt.savefig("./output/temp/fluorescence/recovery.png")

    # # compare about fluorescence recovery
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(sigma_coefs, fluorescence_steady)
    ax.set_xlabel("sigma coefficients")
    ax.set_ylabel("Fluorescence at steady state")

    mkdir("./output/temp/fluorescence/")
    plt.savefig("./output/temp/fluorescence/steady_state.png")

    # # plot each radius
    img_cpy = imgs[2].copy()
    sigma_coefs = np.arange(1, 4.1, 1)
    for sigma_coef in sigma_coefs:
        radius = [s * sigma_coef for s in sigma]
        box = [[center[1], center[0]], [radius[1], radius[0]], 0.0]
        cv2.ellipse(img_cpy, box, color=255, thickness=1)
    cv2.imwrite("./output/temp/illuminance/roi_sigma_samples.png", img_cpy)


def plot_fluorescence(time: list[int], fluorescence_list: list[float]):
    """plot a fluorescence change"""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(time, fluorescence_list, color="tab:green")
    ax.set_xlim([0, 19])
    ax.set_xticks([i for i in range(0, time[-1], 2)])
    ax.set_ylim([0, 100])
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Fluorescence")
    plt.savefig("./output/result/fluorescence.png")


def plot_fluorescence_compare(time: list[int], fluorescence_list: list[list[float]]):
    """plot fluorescence changes"""
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(time, fluorescence_list[0], color="tab:red", label="0.5σ")  # 0.5 sigma
    ax.plot(time, fluorescence_list[1], color="tab:green", label="σ")  # 1 sigma
    ax.plot(time, fluorescence_list[2], color="tab:blue", label="2σ")  # 2 sigma
    ax.set_xlim([0, 19])
    ax.set_ylim([0, 100])
    ax.set_xticks([i for i in range(0, time[-1], 2)])
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Fluorescence")
    ax.legend()
    plt.savefig("./output/result/fluorescence_comp.png")


def plot_fluorescence_fitting(
    time: list[int], fluorescence_list: list[float], result: dict, suffix: ""
):
    A = result["A"]
    K = result["K"]
    C = result["C"]

    step = 0.05
    x = np.arange(time[0], time[-1] + step, step)
    y_pred = ModelFitting.fluorescence_func(x, A, K, C)

    time = time + 2
    x = x + 2

    # plot
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(time, fluorescence_list, color="k")
    ax.plot(x, y_pred, color="b")

    ax.set_xlim([time[0], time[-1]])
    # ax.set_ylim([0, 100])
    ax.set_xticks(time)
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Fluorescence")

    mkdir("./output/temp/fitting")
    if suffix == "":
        plt.savefig("./output/temp/fitting/FRAP.png")
    else:
        plt.savefig("./output/temp/fitting/FRAP_{}.png".format(suffix))


def calc_diffusivity_coefficients(radius, tau):
    D = radius * radius / 4 / tau
    return D


def main():
    img_size = 20

    # load images
    file_paths = ["./HQ/frap{:0>2}.tiff".format(i + 1) for i in range(img_size)]
    imgs = [cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) for file_path in file_paths]

    # find ROI
    # center, sigma = find_roi(imgs)
    center, sigma = find_roi2(imgs)

    # examine_roi(imgs, center, sigma)

    # get fluorescence
    coef_list = [0.5, 1, 2]

    fluorescence_list = []
    fluorescence_ratio_list = []
    for coef in coef_list:
        radius = coef * min(sigma)
        fluore, fluore_ratio = calc_fluorescence(imgs, center, radius)
        fluorescence_list.append(fluore)
        fluorescence_ratio_list.append(fluore_ratio)

    # plot fluorescence change
    dt = 1  # sec
    time = np.arange(img_size) * dt
    plot_fluorescence(time, fluorescence_ratio_list[1])
    plot_fluorescence_compare(time, fluorescence_ratio_list)

    # get diffusivity coefficients
    diff_coef_list = []
    for coef, fluorescence in zip(coef_list, fluorescence_list):
        radius = coef * min(sigma)

        fluorescence = fluorescence[2:]
        time = np.arange(len(fluorescence)) * dt
        result = ModelFitting.diffusion_axelrod1976(time, fluorescence)

        plot_fluorescence_fitting(time, fluorescence, result, str(coef) + "sigma")

        D = calc_diffusivity_coefficients(radius, result["K"])
        diff_coef_list.append(D)

    print(diff_coef_list)


if __name__ == "__main__":
    # make_gif()
    # convert_tiff_to_png()
    main()
    # test()
