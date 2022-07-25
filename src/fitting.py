import lmfit
import matplotlib.pyplot as plt
import numpy as np

from paperplot import plotsetting as ps
from paperplot import plotutils as pu
from utils import mkdir


class ModelFitting:
    @staticmethod
    def gaussian2D(x, y, amplitude, centerx, centery, sigma):
        """オリジナルの2次元ガウス関数モデル"""
        dx, dy = centerx - x, centery - y
        index = -1 * (dx ** 2 + dy ** 2) / (2 * sigma * sigma)
        coef = amplitude / (2 * np.pi * sigma * sigma)
        return coef * np.exp(index)

    @staticmethod
    def gaussian2d_fit(im, report=False):
        """2次元ガウス関数でフィッティング
        独自のモデルを使用. sigmaはx, y軸で共通.

        Args:
            im (numpy.ndarray): フィッティングする画像
            report (bool, optional): 詳細な結果を出力する. Defaults to False.

        Returns:
            tuple: 結果. "中心, シグマ, 半値幅"
        """
        h, w = im.shape[0], im.shape[1]
        z = im.flatten().astype(np.float64)

        x, y = [], []
        for ix in range(h):
            for iy in range(w):
                x.append(ix)
                y.append(iy)
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # create a model and fit
        model = lmfit.Model(ModelFitting.gaussian2D, independent_vars=["x", "y"])

        params = model.make_params()
        params["amplitude"].set(value=128, min=0, vary=True)
        params["centerx"].set(value=h // 2, min=0, max=h, vary=True)
        params["centery"].set(value=w // 2, min=0, max=w, vary=True)
        params["sigma"].set(value=50, min=1, vary=True)

        result = model.fit(z, x=x, y=y, params=params)

        # log result
        if report:
            lmfit.report_fit(result)

        # organizing results
        result_dict = result.params.valuesdict()
        amp = result_dict["amplitude"]
        center = result_dict["centerx"], result_dict["centery"]
        sigma = result_dict["sigma"]
        fwhm = np.sqrt(np.log(2) * 2 * sigma * sigma) * 2

        center = [round(c) for c in center]
        sigma = round(sigma)
        fwhm = round(fwhm)

        # prediction
        z_pred = model.func(x, y, **result.best_values)

        # plot log
        pu.set_base_profiles(plt)

        fig = plt.figure(figsize=(1.5, 1.5), dpi=ps.FigDPI)
        ax = fig.add_subplot(111)
        pu.set_base_axes_profiles(ax)

        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("Luminance (y=center-y)")

        ax.set_xlim([0, w])
        ax.set_ylim([0, 160])
        ax.set_xticks([i * 50 for i in range(4)])
        ax.set_yticks([i * 50 for i in range(4)])

        # # show illuminance change
        x = [i for i in range(im.shape[1])]
        y = im[center[0], :]
        ax.plot(x, y, color="k", linewidth=ps.LineWidth, label="actual")

        # # show prediction
        z_pred = z_pred.reshape(h, w)
        y_pred = z_pred[center[0], :]
        ax.plot(x, y_pred, color="b", linewidth=ps.LineWidth, label="model")

        fig_path = "./output/temp/fitting/gaussian"
        fig_name = "tomography"
        pu.save_fig2(fig, fig_path + "/jpg", fig_name, "jpg")
        pu.save_fig2(fig, fig_path + "/pdf", fig_name, "pdf")

        plt.close(fig)

        return center, sigma, fwhm

    @staticmethod
    def quadratic2D(x, y, centerx, centery, amp, const):
        dx, dy = x - centerx, y - centery
        z = amp * (dx ** 2 + dy ** 2) + const
        z[z < 0] = 0
        return z

    @staticmethod
    def quadratic2d_fit2(im, report=False):
        """2変数の二次関数様関数でフィッティング
        独自のモデルを使用.

        Args:
            im (numpy.ndarray): フィッティングする画像
            report (bool, optional): 詳細な結果を出力する. Defaults to False.

        Returns:
            tuple: 結果. "中心, 半径"
        """
        h, w = im.shape[0], im.shape[1]
        z = im.flatten().astype(np.float64)

        x, y = [], []
        for ix in range(h):
            for iy in range(w):
                x.append(ix)
                y.append(iy)
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        # create a model and fit
        model = lmfit.Model(ModelFitting.quadratic2D, independent_vars=["x", "y"])

        params = model.make_params()
        params["centerx"].set(value=h // 2, min=0, max=h, vary=True)
        params["centery"].set(value=w // 2, min=0, max=w, vary=True)
        params["amp"].set(value=-10, max=0, vary=True)
        params["const"].set(value=128, min=0, max=255, vary=True)

        result = model.fit(z, x=x, y=y, params=params)

        # log result
        if report:
            lmfit.report_fit(result)

        # organizing results
        result_dict = result.params.valuesdict()
        center = [round(result_dict["centerx"]), round(result_dict["centery"])]
        a = result_dict["amp"]
        c = result_dict["const"]

        # calc radius (z = 0)
        radius = np.sqrt(-c / a)
        radius = round(radius)

        # prediction
        z_pred = model.func(x, y, **result.best_values)

        # plot log
        pu.set_base_profiles(plt)

        fig = plt.figure(figsize=(1.5, 1.5), dpi=ps.FigDPI)
        ax = fig.add_subplot(111)
        pu.set_base_axes_profiles(ax)

        ax.set_xlabel("x [pixel]")
        ax.set_ylabel("Luminance (y=center-y)")

        ax.set_xlim([0, w])
        ax.set_ylim([0, 160])
        ax.set_xticks([i * 50 for i in range(4)])
        ax.set_yticks([i * 50 for i in range(4)])

        # # show illuminance change
        x = [i for i in range(im.shape[1])]
        y = im[center[0], :]
        ax.plot(x, y, color="k", linewidth=ps.LineWidth, label="actual")

        # # show prediction
        z_pred = z_pred.reshape(h, w)
        y_pred = z_pred[center[0], :]
        ax.plot(x, y_pred, color="b", linewidth=ps.LineWidth, label="model")

        fig_path = "./output/temp/fitting/quadratic"
        fig_name = "tomography_quadratic"
        pu.save_fig2(fig, fig_path + "/jpg", fig_name, "jpg")
        pu.save_fig2(fig, fig_path + "/pdf", fig_name, "pdf")

        plt.close(fig)

        return center, radius

    @staticmethod
    def fluorescence_func(t: float, A: float, K: float, C: float):
        return A * (1 - np.exp(-1 * t / K)) + C

    @staticmethod
    def diffusion_axelrod1976(time: list[float], fluorescence: list[float]):
        # create a model and fit
        model = lmfit.Model(ModelFitting.fluorescence_func, independent_vars=["t"])

        params = model.make_params()
        params["A"].set(value=10, vary=True)
        params["K"].set(value=10, vary=True)
        params["C"].set(value=fluorescence[0], vary=False)

        result = model.fit(fluorescence, t=time, params=params)

        # organize results
        result_dict = result.params.valuesdict()
        return result_dict
