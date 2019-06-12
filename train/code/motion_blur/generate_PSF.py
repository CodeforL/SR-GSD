import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from motion_blur.generate_trajectory import Trajectory
from PIL import Image
from scipy import signal
from scipy.ndimage import convolve


class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas,max_len=canvas*0.8, expl=0.003).fit(show=False, save=False).x
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
            #self.fraction = [1 , 1 , 1 , 1]
        else:
            self.fraction = fraction
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, np.math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, np.math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))
        if show or save:
            self.__plot_canvas(show, save)

        return self.PSFs

    def __plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            plt.close()
            fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))

            for i in range(self.PSFnumber):
                axes[i].imshow(self.PSFs[i], cmap='gray')
                print(self.PSFs[i].shape)
                print(self.PSFs[i].sum())
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()


def plot(kernel,can):
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    x = np.arange(can)
    y = np.arange(can)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.plot_wireframe(X,Y,kernel,color='c')
    # plt.show()
    # 曲面，x,y,z坐标，横向步长，纵向步长，颜色，线宽，是否渐变
    # cm.coolwarm
    surf = ax.plot_surface(X, Y, kernel, rstride=1, cstride=1, cmap='plasma', edgecolor='none')
    plt.show()


if __name__ == '__main__':
    can = 15
    motion_blur_list = []
    params = [0.01, 0.009, 0.008, 0.007,0.006, 0.005,0.004, 0.003,0.002]
    for p in params:
        for i in range(10):
            trajectory = Trajectory(canvas=can, max_len=can*0.8, expl=p).fit()
            psf = PSF(canvas=can, trajectory=trajectory).fit()
            motion_blur_list.append(np.reshape(psf[3],(1,can,can)))
        print("w")

    motion_blur_list = np.concatenate(motion_blur_list,axis=0)
    print(motion_blur_list.shape)
    save_path = "motion_blur_kernel.npy"
    np.save(save_path,motion_blur_list)


'''
    img  = Image.open('0001.png')
    #img.show()
    img = Image.fromarray(convolve(img, np.reshape(psf.PSFs[3],(can,can,1)), mode='nearest'))
    img.show()
'''
