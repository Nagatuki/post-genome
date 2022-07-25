from __future__ import annotations

import os

import libs.paperplot.plotsetting as ps
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def set_base_profiles(plt):
    plt.rcParams["font.family"] = ps.Font.CALIBRI  # 使用するフォント
    plt.rcParams["xtick.direction"] = "in"  # x軸の目盛線を内向き('in')
    plt.rcParams["ytick.direction"] = "in"  # y軸の目盛線を内向き('in')
    plt.rcParams["font.size"] = 9  # フォントの大きさ
    plt.rcParams["axes.linewidth"] = 0.8  # 軸の線幅edge linewidth。囲みの太さ
    plt.rcParams["xtick.major.width"] = 0.6  # x軸主目盛り線の線幅
    plt.rcParams["ytick.major.width"] = 0.6  # y軸主目盛り線の線幅

    plt.rcParams["pdf.fonttype"] = 42  # PDFで出力した際
    plt.rcParams["ps.fonttype"] = 42

    plt.gca().spines["top"].set_visible(False)  # プロットの上の線を消す
    plt.gca().spines["right"].set_visible(False)  # プロットの右の線を消す


def set_base_axes_profiles(ax):
    ax.spines["top"].set_visible(False)  # プロットの上の線を消す
    ax.spines["right"].set_visible(False)  # プロットの右の線を消す


def set_font(plt, font: str):
    plt.rcParams["font.family"] = font


def set_font_size(plt, font_size: int):
    plt.rcParams["font.size"] = font_size


def set_tick_labelsize(plt, size: int):
    plt.rcParams["xtick.labelsize"] = size
    plt.rcParams["ytick.labelsize"] = size


def save_fig(fig: figure, file_name: str):
    """画像を保存する.
    matplotlib特有の謎の余白を減らすことができる.
    Args:
        fig (matplotlib.pyplot.figure): fig
        file_name (str): ファイル名
    """
    fig.savefig(file_name, bbox_inches="tight", pad_inches=0.05, transparent=True)


def save_fig2(fig: figure, dir_path: str, file_name: str, file_type: str):
    """画像を保存する
    save_figとほぼ同じだが, こちらはディレクトリパス, ファイル名, ファイル形式を指定する.
    Args:
        fig (figure): fig
        dir_path (str): 保存先のディレクトリ
        file_name (str): ファイル名(ファイル形式をつけてはいけない)
        file_type (str): ファイル形式(ex. png, jpg, eps)
    """
    # create directory if not exsist
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # set file name
    file_name = "{}/{}.{}".format(dir_path, file_name, file_type)

    # save
    save_fig(fig, file_name)


def create_colors_from_cmap(name: str, size: int) -> list:
    """matplotlibのカラーマップから色の配列を取得する
    同系統の色のリストが欲しい時に使用する。
    Args:
        name (str): カラーマップの名前
        size (int): カラーリストの要素数
    Returns:
        list: カラーリスト
    """
    color_idx = [(idx + 1) / (size + 1) for idx in range(size)]

    cmap = plt.get_cmap(name)
    colors = [cmap(idx) for idx in color_idx]

    return colors
