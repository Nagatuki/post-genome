import cv2
import matplotlib.pyplot as plt
import numpy as np


def moving_average(im, ksize=3):
    kernel = np.full((ksize, ksize), 1 / (ksize ** 2))
    ret_im = cv2.filter2D(im, -1, kernel)
    return ret_im


def weighted_average(im, ksize=3, sigma=2):
    ret_im = cv2.GaussianBlur(img, (ksize, ksize), sigma, sigma)
    return ret_im


def median(im, ksize=3):
    ret_im = cv2.medianBlur(im, ksize=ksize)
    return ret_im


def sobel_filter(im, ksize=1):
    dst1 = cv2.Sobel(img, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=ksize)
    dst2 = cv2.Sobel(img, ddepth=cv2.CV_16S, dx=0, dy=1, ksize=ksize)
    ret_im = np.sqrt(dst1 ** 2 + dst2 ** 2)
    return ret_im


def gaussian_filter(im, ksize=3, sigma=1.3):
    ret_im = cv2.GaussianBlur(im, ksize=(ksize, ksize), sigmaX=sigma)
    return ret_im


class FrequencyFilter:
    @staticmethod
    def ideal_mask(im, R, inv=False):
        """円形のマスク画像を作成"""
        h, w = im.shape[0], im.shape[1]
        c_h, c_w = h // 2, w // 2

        mask = np.zeros([h, w])
        for x in range(h):
            for y in range(w):
                distance = (x - c_h) ** 2 + (y - c_w) ** 2
                if distance < R * R:
                    mask[x][y] = 1
                else:
                    mask[x][y] = 0

        if inv:
            mask = (mask * -1) + 1

        return mask

    @staticmethod
    def gaussian_mask(im, sigma):
        h, w = im.shape[0], im.shape[1]
        c_h, c_w = h // 2, w // 2

        mask = np.zeros([h, w])
        A = 1 / (2 * np.pi * sigma * sigma)
        for x in range(h):
            for y in range(w):
                dx = x - c_h
                dy = y - c_w
                in_exp = -1 * (dx * dx + dy * dy) / (2 * (sigma ** 2))
                mask[x][y] = A * np.exp(in_exp)

        return mask

    @staticmethod
    def lowpass(im, fc, mask_type):
        im_fft = np.fft.fft2(im)
        im_shift = np.fft.fftshift(im_fft)

        if mask_type == "ideal":
            mask = FrequencyFilter.ideal_mask(im, fc, inv=False)
        if mask_type == "gauss" or mask_type == "gaussian":
            mask = None
        im_fft_masked = np.multiply(im_shift, mask)

        im_fft_ret = np.fft.fftshift(im_fft_masked)
        im_ret = np.fft.ifft2(im_fft_ret)
        return np.uint8(im_ret.real)

    @staticmethod
    def highpass(im, fc, mask_type):
        im_fft = np.fft.fft2(im)
        im_shift = np.fft.fftshift(im_fft)

        if mask_type == "ideal":
            mask = FrequencyFilter.ideal_mask(im, fc, inv=True)
        if mask_type == "gauss" or mask_type == "gaussian":
            mask = None
        im_fft_masked = np.multiply(im_shift, mask)

        im_fft_ret = np.fft.fftshift(im_fft_masked)
        im_ret = np.fft.ifft2(im_fft_ret)
        return np.uint8(im_ret.real)

    @staticmethod
    def bandpass(im, fc_low, fc_high, mask_type):
        im_fft = np.fft.fft2(im)
        im_shift = np.fft.fftshift(im_fft)

        if mask_type == "ideal":
            mask_low = FrequencyFilter.ideal_mask(im, fc_high, inv=False)
            mask_high = FrequencyFilter.ideal_mask(im, fc_low, inv=True)
        if mask_type == "gauss" or mask_type == "gaussian":
            mask_low = None
            mask_high = None
        mask = np.multiply(mask_low, mask_high)
        im_fft_masked = np.multiply(im_shift, mask)

        im_fft_ret = np.fft.fftshift(im_fft_masked)
        im_ret = np.fft.ifft2(im_fft_ret)

        plt.figure()
        plt.imshow(np.uint8(im_ret.real))
        plt.gray()
        plt.show()

        return np.uint8(im_ret.real)


def test():
    file_path = "./sample/01.png"
    im = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    filtered = filter.FrequencyFilter.highpass(im, 1, "ideal")
    cv2.imwrite("./output/sample_output.png", filtered)
