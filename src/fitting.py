import lmfit
import matplotlib.pyplot as plt
import numpy as np

from utils import mkdir


class ModelFitting:
    @staticmethod
    def gaussian2d_fit(im, report=False):
        """2次元ガウス関数でフィッティング
        lmfitで実装済みのモデルを使用. sigmaはx, y軸で別.

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
        model = lmfit.models.Gaussian2dModel()

        params = model.make_params()
        params["amplitude"].set(value=128, min=0, vary=True)
        params["centerx"].set(value=h // 2, min=0, max=h, vary=True)
        params["centery"].set(value=w // 2, min=0, max=w, vary=True)
        params["sigmax"].set(value=50, min=5, vary=True)
        params["sigmay"].set(value=50, min=5, vary=True)

        result = model.fit(z, x=x, y=y, params=params)

        # log result
        if report:
            lmfit.report_fit(result)

        # organizing results
        result_dict = result.params.valuesdict()
        amp = result_dict["amplitude"]
        center = result_dict["centerx"], result_dict["centery"]
        sigma = result_dict["sigmax"], result_dict["sigmay"]
        fwhm = result_dict["fwhmx"], result_dict["fwhmy"]

        amp = round(amp)
        center = [round(c) for c in center]
        sigma = [round(s) for s in sigma]
        fwhm = [round(f) for f in fwhm]

        # prediction
        z_pred = model.func(x, y, **result.best_values)

        # # show illuminance change
        x = [i for i in range(im.shape[1])]
        y = im[center[0], :]
        plt.plot(x, y, color="k")

        # # show prediction
        z_pred = z_pred.reshape(h, w)
        y_pred = z_pred[center[0], :]
        plt.plot(x, y_pred, color="b")

        mkdir("./output/temp/fitting")
        plt.savefig("./output/temp/fitting/tomography.png")

        return center, sigma, fwhm

    @staticmethod
    def gaussian2D(x, y, amplitude, centerx, centery, sigma):
        """オリジナルの2次元ガウス関数モデル"""
        dx, dy = centerx - x, centery - y
        index = -1 * (dx ** 2 + dy ** 2) / (2 * sigma * sigma)
        coef = amplitude / (2 * np.pi * sigma * sigma)
        return coef * np.exp(index)

    @staticmethod
    def gaussian2d_fit2(im, report=False):
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
        fwhm = np.sqrt(np.log(2) * 2 * sigma * sigma)

        center = [round(c) for c in center]
        sigma = [round(sigma), round(sigma)]
        fwhm = [round(fwhm), round(fwhm)]

        # prediction
        z_pred = model.func(x, y, **result.best_values)

        # # show illuminance change
        x = [i for i in range(im.shape[1])]
        y = im[center[0], :]
        plt.plot(x, y, color="k")

        # # show prediction
        z_pred = z_pred.reshape(h, w)
        y_pred = z_pred[center[0], :]
        plt.plot(x, y_pred, color="b")

        mkdir("./output/temp/fitting")
        plt.savefig("./output/temp/fitting/tomography.png")

        return center, sigma, fwhm

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
