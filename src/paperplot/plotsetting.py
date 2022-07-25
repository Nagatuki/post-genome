from __future__ import annotations


class FigWidth:
    # 横幅 180 mm
    SINGLE_COLUMN = 7.09

    # 横幅 85 mm
    DOUBLE_COLUMN = 3.35


class FigAspectRatio:
    # 4 x 3
    RATIO_4x3 = (4, 3)

    # 16 x 9
    RATIO_16x9 = (16, 9)

    # 1 x 1
    RATIO_1x1 = (1, 1)


class FigSize:
    def __init__(self, width, height):
        self.__width = width
        self.__height = height

    def get(self) -> tuple(float, float):
        return (self.__width, self.__height)

    @classmethod
    def single_column(cls, aspect_ratio=FigAspectRatio.RATIO_4x3):
        width = FigWidth.SINGLE_COLUMN
        height = width * aspect_ratio[1] / aspect_ratio[0]
        return FigSize(width, height)

    @classmethod
    def double_column(cls, aspect_ratio=FigAspectRatio.RATIO_4x3):
        width = FigWidth.DOUBLE_COLUMN
        height = width * aspect_ratio[1] / aspect_ratio[0]
        return FigSize(width, height)

    @classmethod
    def original_column(cls, width: float, aspect_ratio=FigAspectRatio.RATIO_4x3):
        return cls.original(width=width, aspect_ratio=aspect_ratio)

    @classmethod
    def original(cls, width: float, aspect_ratio=FigAspectRatio.RATIO_4x3):
        height = width * aspect_ratio[1] / aspect_ratio[0]
        return FigSize(width, height)


FigDPI = 300  # figure dpi
LineWidth = 0.8  # widht of line plot


class Font:
    # 通常文字
    ARIAL = "arial"
    CALIBRI = "calibri"
    TIMES_NEW_ROMAN = "times new roman"

    # 数式
    # # 通常文字と相性のいい数式フォント(サンセリフ)
    STIXSANS = "stixsans"

    # # TeXと同じフォント？
    COMPUTER_MODERN = "cm"

    # 現在の全体フォント設定
    CURRENT_FONT = "arial"

    @classmethod
    def set_font(cls, font: str):
        cls.CURRENT_FONT = font

    @classmethod
    def get_font(cls) -> str:
        return cls.CURRENT_FONT


class ErrorBar:
    elinewidth = 0.6
    capsize = 2
    capthick = 0.6

    @classmethod
    def get_kwargs(cls):
        kwargs = {
            "elinewidth": cls.elinewidth,
            "capsize": cls.capsize,
            "capthick": cls.capthick,
        }
        return kwargs
