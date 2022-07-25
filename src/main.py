import cv2
import lmfit
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton

import imfilter
from fitting import ModelFitting
from paperplot import plotsetting as ps
from paperplot import plotutils as pu
from utils import mkdir


def denoise(imgs: list[np.ndarray], log: bool = False) -> list[np.ndarray]:
    """画像のノイズ除去
    フィルタサイズ5の移動平均フィルタを適用する.

    Args:
        imgs (list[np.ndarray]): 画像群

    Returns:
        list[np.ndarray]: 処理後の画像群
    """
    ksize = 7  # フィルタサイズ
    # filtered_imgs = [imfilter.moving_average(img, ksize=ksize) for img in imgs]
    filtered_imgs = [imfilter.moving_average(img, ksize=ksize) for img in imgs]

    if log:
        dir_path = "./output/temp/denoise"
        mkdir(dir_path)

        for i, img in enumerate(filtered_imgs):
            file_name = dir_path + "/filtered{:0>2}.png".format(i + 1)
            cv2.imwrite(file_name, img)

    return filtered_imgs


def find_roi(imgs: list[np.ndarray], log: bool = False) -> tuple[tuple, float]:
    """ROIの算出
    ROIの検索にはガウス関数のフィッティングを行っている.

    Args:
        imgs (list[np.ndarray]): 画像群
        log (bool, optional): 解析結果のログ出力の有無. Defaults to False.

    Returns:
        tuple[tuple, float]: 算出されたROI. "中心点, 半径"
    """
    # choose before and after bleaching figures
    im_pre, im_post = imgs[1], imgs[2]

    # get fluorescence changes
    im_diff = im_pre - im_post

    # gaussian fitting
    center, sigma, fwhm = ModelFitting.gaussian2d_fit(im_diff, report=True)

    # determine bleaching radius
    # radius = sigma  # 1SD
    radius = sigma * 2  # 2SD
    # radius = fwhm / 2  # hwhm

    if log:
        # output dir
        dir_path = "./output/temp/find_roi"
        mkdir(dir_path)

        # fluorescence changes
        cv2.imwrite(dir_path + "/im_diff.png", im_diff)

        # fitting result
        im_diff_cpy = im_diff.copy()
        box = [[center[1], center[0]], [2 * sigma, 2 * sigma], 0.0]
        cv2.ellipse(im_diff_cpy, box, color=255, thickness=1)
        cv2.imwrite(dir_path + "/found_roi_sigma.png", im_diff_cpy)

        im_diff_cpy = im_diff.copy()
        box = [[center[1], center[0]], [4 * sigma, 4 * sigma], 0.0]
        cv2.ellipse(im_diff_cpy, box, color=255, thickness=1)
        cv2.imwrite(dir_path + "/found_roi_2sigma.png", im_diff_cpy)

        im_diff_cpy = im_diff.copy()
        box = [[center[1], center[0]], [fwhm, fwhm], 0.0]
        cv2.ellipse(im_diff_cpy, box, color=255, thickness=1)
        cv2.imwrite(dir_path + "/found_roi_fwhm.png", im_diff_cpy)

    return center, radius


def find_roi_quadratic(imgs, log: bool = False) -> tuple[tuple, float]:
    """ROIの算出
    ROIの検索には二次関数様の関数でフィッティングを行っている.

    Args:
        imgs (list[np.ndarray]): 画像群
        log (bool, optional): 解析結果のログ出力の有無. Defaults to False.

    Returns:
        tuple[tuple, float]: 算出されたROI. "中心点, 半径"
    """
    # choose before and after bleaching figures
    im_pre, im_post = imgs[1], imgs[2]

    # get fluorescence changes
    im_diff = im_pre - im_post

    # quadratic fitting
    center, radius = ModelFitting.quadratic2d_fit2(im_diff, True)

    if log:
        # output dir
        dir_path = "./output/temp/find_roi_quadratic"
        mkdir(dir_path)

        # fluorescence changes
        cv2.imwrite(dir_path + "/im_diff.png", im_diff)

        # check fitting result
        im_diff_cpy = im_diff.copy()
        cv2.circle(im_diff_cpy, [center[1], center[0]], radius, color=255, thickness=1)
        cv2.imwrite(dir_path + "/found_roi_radius.png", im_diff_cpy)

    return center, radius


def calc_fluorescence(
    imgs: np.ndarray, center: tuple[int], radius: float
) -> list[float]:
    """画像ROI中の蛍光強度変化を導出

    Args:
       imgs (np.ndarray): 画像群
        center (tuple[int]): ROI中心
        radius (float): ROI半径

    Returns:
        list[float]: 蛍光強度の変遷
    """
    fluorescence_list = []

    for img in imgs:
        sum_fluorescence = 0
        count_pixel = 0
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                dx = x - center[0]
                dy = y - center[1]

                if dx ** 2 + dy ** 2 <= radius ** 2:
                    sum_fluorescence += img[x, y]
                    count_pixel += 1

        fluorescence_list.append(sum_fluorescence / count_pixel)

    return fluorescence_list


def plot_fluorescence(
    time: list[int], fluorescence_list: list[float], fig_path: str, fig_name: str
):
    """plot a fluorescence change"""
    pu.set_base_profiles(plt)

    fig = plt.figure(figsize=(2, 1.5), dpi=ps.FigDPI)
    ax = fig.add_subplot(111)
    pu.set_base_axes_profiles(ax)

    ax.plot(time, fluorescence_list, linewidth=ps.LineWidth, color="tab:green")
    ax.set_xlim([0, 19])
    ax.set_xticks([i * 2 for i in range(10)])
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Fluorescence intensity")

    pu.save_fig2(fig, fig_path + "/jpg", fig_name, "jpg")
    pu.save_fig2(fig, fig_path + "/pdf", fig_name, "pdf")


def calc_diffusivity_uniform_beam(fluorescence: list[float], radius: float):
    fluorescence = fluorescence[2:]
    time = np.arange(0, len(fluorescence), 1)

    result = ModelFitting.fluorescence_fitting(time, fluorescence)
    plot_fluorescence_fitting(time, fluorescence, result)

    print("Fluorescence recover fitting")
    print(result)

    T = result["T"]
    tau_half = np.log(2) * T

    radius = radius * 0.225  # pixel to um
    D = 0.224 * radius * radius / tau_half
    return D


# def calc_diffusivity_coefficients(fluorescence: list[float], radius: float):
#     # get K (the bleaching parameter, depends on laser characteristic)
#     # use Newton's method
#     FKM, FK0 = fluorescence[1], fluorescence[2]
#     A = FK0 / FKM
#     func = lambda K, A: A * K + np.exp(-K) - 1
#     func_prime = lambda K, A: A - np.exp(-K)
#     K = newton(func, 100, fprime=func_prime, args=(A))
#     print(K)
#     pass


def plot_fluorescence_fitting(time: list[int], fluorescence: list[float], result: dict):
    A = result["A"]
    K = result["T"]
    C = result["C"]

    step = 0.05
    x = np.arange(time[0], time[-1] + step, step)
    y_pred = ModelFitting.fluorescence_func(x, A, K, C)

    time = time + 2
    x = x + 2

    # plot
    pu.set_base_profiles(plt)

    fig = plt.figure(figsize=(2, 1.5), dpi=ps.FigDPI)
    ax = fig.add_subplot(111)
    pu.set_base_axes_profiles(ax)

    ax.scatter(time, fluorescence, color="k", s=4)
    ax.plot(x, y_pred, linewidth=ps.LineWidth, color="b")
    ax.set_xlim([2, 20])
    ax.set_xticks([2 + i * 2 for i in range(9)])
    ax.set_ylim([100, 120])
    ax.set_yticks([100 + i * 5 for i in range(5)])
    ax.set_xlabel("Time [sec]")
    ax.set_ylabel("Fluorescence intensity")

    fig_path = "./output/temp/fitting"
    fig_name = "FRAP"
    pu.save_fig2(fig, fig_path + "/jpg", fig_name, "jpg")
    pu.save_fig2(fig, fig_path + "/pdf", fig_name, "pdf")


def main():
    img_size = 20

    # load images
    file_paths = ["./HQ/frap{:0>2}.tiff".format(i + 1) for i in range(img_size)]
    imgs = [cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) for file_path in file_paths]

    # denoise
    imgs = denoise(imgs, log=True)

    # find ROI
    center, radius = find_roi(imgs, True)
    # center_q, radius_q = find_roi_quadratic(imgs, False)

    # examine_roi(imgs, center, sigma)

    # get fluorescence change
    time = np.arange(img_size) * 1  # dt = 1
    fluorescence = calc_fluorescence(imgs, center, radius)
    # fluorescence_q = calc_fluorescence(imgs, center_q, radius_q)

    # # plot fluorescence change
    plot_fluorescence(time, fluorescence, "./output/result/gaussian", "fluorescence")
    # plot_fluorescence(time, fluorescence_q, "./output/result/quadratic", "fluorescence")

    # # get diffusivity coefficients
    D = calc_diffusivity_uniform_beam(fluorescence, radius)
    print(D)


if __name__ == "__main__":
    # from utils import make_gif
    # make_gif()

    # from utils import convert_tiff_to_png
    # convert_tiff_to_png()

    # test()

    main()
