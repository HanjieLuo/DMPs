# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.metrics import mean_squared_error
import time


class Canonical(object):
    def __init__(self, dt, kernel_num, ax):
        self.dt = dt
        self.ax = ax
        self.kernel_num = kernel_num
        self.x = 1.0

    # def init(self, run_time):
    #     self.run_time = run_time

    def run(self, t, tau=None):
        if tau is None:
            # 如果tau为None, t表示为归一化后的时间
            return np.exp(- self.ax * t)
        else:
            # 否则t表示具体的时间
            return np.exp(- self.ax * t / tau)

    # def run_discrete(self):
    #     self.x += (- self.ax * self.x) * self.tau * self.dt
    #     return self.x

    # def reset_discrete(self):
    #     self.x = 1.0


class DMP(object):
    def __init__(self, system_dt=0.001):
        self.kernel_num = 150
        self.az = 48  # 48
        self.bz = self.az / 4
        # self.ax = abs(np.log(0.0000001))
        self.ax = 1
        self.tau = 1

        self.t = 0.0
        self.cur_step = 0

        self.system_dt = system_dt
        self.internal_dt = min(system_dt, 0.001)
        self.steps_between_measurement = int(self.system_dt / self.internal_dt)

        self.canonical = Canonical(self.internal_dt, self.kernel_num, self.ax)
        self.h = np.zeros(self.kernel_num)

    def init(self, paths, paths_v, paths_a, run_time, debug=False):
        self.reset()

        self.tau = run_time
        self.joints_num = len(paths)
        self.steps = int(run_time / self.internal_dt)

        self.az = self.az * np.ones(self.joints_num)
        self.bz = self.bz * np.ones(self.joints_num)

        self.goals = paths[:, -1]
        self.y0 = paths[:, 0].copy()

        self.y = self.y0.copy()
        self.y_v = np.zeros(self.joints_num)
        self.y_a = np.zeros(self.joints_num)
        self.scalar = self.goals - self.y0

        self.check_offset()

        self.gen_centers()
        # print "centers", self.centers
        # print "h", self.h

        self.f_target = self.gen_force_target(path, path_v, path_a)


        if debug is True:
            # print "c", self.centers
            # print "h", self.h
            t = np.linspace(0.0, self.tau, self.steps)
            x = self.canonical.run(t, self.tau)
            psi = self.gen_psi(x)
            # plt.plot(t, x, 'b-')
            # plt.plot(np.zeros(self.kernel_num))
            for i in range(self.kernel_num):
                plt.plot(t, psi[i, :], '--')
            plt.show()
        # print self.f_target
        # t = np.linspace(0.0, self.tau, self.steps)
        # for i in range(self.joints_num):
        #     plt.plot(t, self.f_target[i, :], '--')
        # plt.show()

        self.weights = self.gen_weights(paths)
        # print self.weights
        # # print the centers

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""
        for d in range(self.joints_num):
            if (self.y0[d] == self.goals[d]):
                self.goal[d] += 1e-4

    def reset(self):
        self.t = 0.0
        self.cur_step = 0

    def gen_force_target(self, paths, paths_v, paths_a):
        _, path_steps = paths.shape
        if path_steps != self.steps:
            from scipy import interpolate

            # generate function to interpolate the desired trajectory
            t_ori = np.linspace(0.0, self.tau, path_steps)
            t_new = np.linspace(0.0, self.tau, self.steps)

            paths_gen = np.zeros((self.joints_num, self.steps))
            paths_v_gen = np.zeros((self.joints_num, self.steps))
            paths_a_gen = np.zeros((self.joints_num, self.steps))
            for i in range(self.joints_num):
                fun = interpolate.interp1d(t_ori, paths[i, :])
                paths_gen[i, :] = fun(t_new)

                fun = interpolate.interp1d(t_ori, paths_v[i, :])
                paths_v_gen[i, :] = fun(t_new)

                fun = interpolate.interp1d(t_ori, paths_a[i, :])
                paths_a_gen[i, :] = fun(t_new)
        else:
            paths_gen = paths
            paths_v_gen = paths_v
            paths_a_gen = paths_a

        f_target = np.zeros((self.joints_num, self.steps))

        # t = np.linspace(0.0, self.tau, self.steps)
        # for i in range(self.joints_num):
        #     plt.plot(t, paths_gen[0, :], 'r--')
        #     plt.plot(t, paths_v_gen[0, :], 'g--')
        #     plt.plot(t, paths_a_gen[0, :], 'b--')
        # plt.show()

        for d in range(self.joints_num):
            f_target[d, :] = paths_a_gen[d, :] * (self.tau ** 2) - self.az[d] * (self.bz[d] * (self.goals[d] - paths_gen[d, :]) - paths_v_gen[d, :] * self.tau)
        return f_target

    def gen_force(self, x):
        psi = self.gen_psi(x).ravel()
        psi_sum = np.sum(psi)
        f = np.zeros(self.joints_num)
        for i in range(self.joints_num):
            f[i] = np.sum(psi * self.weights[i, :]) * x * self.scalar[i] / psi_sum
            # print "j", i
            # print np.sum(psi * self.weights[i, :])
            # print x
            # print self.scalar[i]
            # print psi_sum
            # print np.sum(psi * self.weights[i, :]) * x * self.scalar[i] / psi_sum
            # print
        return f

    def gen_weights(self, paths):
        t = np.linspace(0.0, self.tau, self.steps)
        x = self.canonical.run(t, self.tau)
        # data = pickle.load(open('data.txt', 'rb'))
        # x = data['x_track']
        w = np.zeros((self.joints_num, self.kernel_num))
        psi = self.gen_psi(x)

        # for i in range(self.kernel_num):
        #     plt.plot(t, psi[i, :])
        # plt.plot(t, x, 'r--')
        # plt.plot(t, self.f_target[0, :])
        # plt.show()
        # print "self.f_target"
        # print self.f_target
        # print
        # print "self.centers", self.centers
        # print

        for k in range(self.kernel_num):
            psii = psi[k, :]
            for j in range(self.joints_num):
                f = self.f_target[j, :]
                s = x * self.scalar[j]
                # print np.sum(s * psi * y)
                w[j, k] = np.sum(s * psii * f) / np.sum(s * s * psii)
                # print j, i, w[j, i]
        # print w
        # print w.shape
        return w

    def gen_psi(self, x):
        if isinstance(x, (int, float)):
            ll = 1
        else:
            ll = len(x)
        psi = np.zeros((self.kernel_num, ll))
        for i in range(self.kernel_num):
            psi[i] = np.exp(- self.h[i] * ((x - self.centers[i]) ** 2))
        return psi

    def gen_centers(self):
        # last = self.canonical.run(self.tau, self.tau)

        centers_time = np.linspace(0., 1., self.kernel_num)
        self.centers = self.canonical.run(centers_time)
        self.h = self.kernel_num**1.5 / (self.centers)
        # print "h", self.h

        # self.centers = np.linspace(0.0, 1.0, self.kernel_num)
        # self.h = self.kernel_num**1.5 / self.centers

        # centers_time = np.linspace(0., 1., self.kernel_num)
        # self.centers = self.canonical.run(centers_time)
        # print "centers", self.centers
        # print "diff", np.diff(self.centers)
        # print "all", np.fabs(0.05 * np.diff(self.centers))
        # self.h[:-1] = np.fabs(0.05 * np.diff(self.centers))
        # self.h[:-1] = np.fabs(0.05 * np.diff(self.centers))
        # self.h[-1] = self.h[-2]
        # print "h", self.h

        # activation = 0.1
        # self.centers = np.linspace(0, 1.0, self.kernel_num)
        # diff = float(1.0) / float(self.kernel_num - 1)

        # print -pow(diff / 2.0, 2) / np.log(activation)
        # self.h[:] = -pow(diff / 2.0, 2) / math.log(activation)
        # print "centers", centers
        # print "self.h", self.h
        # return centers

    def run(self):
        if (self.t > self.tau):
            return

        while True:
            self.y += self.y_v * self.internal_dt
            self.y_v += self.y_a * self.internal_dt

            x = self.canonical.run(self.t, self.tau)
            f = self.gen_force(x)
            # print "x", x
            # print "f", f
            # print
            self.y_a = (self.az * (self.bz * (self.goals - self.y) - self.tau * self.y_v) + f) / (self.tau ** 2)

            self.t += self.internal_dt
            self.cur_step += 1

            if self.cur_step % self.steps_between_measurement == 0:
                break
        return self.y, self.y_v, self.y_a

        # psi = self.gen_psi(x).ravel()
        # force = np.zeros(self.num_joints)
        # print "x", x
        # print "w", self.weights
        # print "psi", psi

        # for i in range(self.num_joints):
        #     force = np.sum(self.weights[i, :] * psi) / np.sum(psi)
        #     self.y_a[i] = self.tau * (self.az[i] * (self.bz[i] * (self.goals[i] - self.y[i]) - self.y_v[i] / self.tau) + force)
        #     self.y_v[i] += self.y_a[i] * self.tau * self.dt
        #     self.y[i] += self.y_v[i] + self.dt

        # return self.y
        #     self.ddy[d] = (self.ay[d] *
        #                    (self.by[d] * (self.goal[d] - self.y[d]) -
        #                    self.dy[d]/tau) + f) * tau
        #     if external_force is not None:
        #         self.ddy[d] += external_force[d]
        #     self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
        #     self.y[d] += self.dy[d] * self.dt * error_coupling
        # print force


if __name__ == "__main__":
    mytime = 8
    dt = 0.01
    path1 = np.sin(np.arange(0, mytime, dt) * 1)
    path2 = np.cos(np.arange(0, mytime, dt) * 1)
    path = np.vstack((path1, path2))

    path_v = np.diff(path) / dt
    # add zero to the beginning of every row
    path_v = np.hstack((np.zeros((2, 1)), path_v))
    # print "path_v", path_v

    path_a = np.diff(path_v) / dt
    path_a = np.hstack((np.zeros((2, 1)), path_a))
    # print "path_a", path_a

    dmp = DMP(dt)
    start_time = time.time()
    dmp.init(path, path_v, path_a, mytime, False)
    print("--- %s seconds ---" % (time.time() - start_time))

    t = np.arange(0, mytime, dt)

    fig, ax = plt.subplots()
    ax.plot(t, path[0], 'b', label='target')
    ax.plot(t, path[1], 'r')

    y = np.empty_like(path)
    y_v = np.empty_like(path_v)
    y_a = np.empty_like(path_a)

    for i in range(int(mytime / dt)):
        y[:, i], y_v[:, i], y_a[:, i] = dmp.run()

    np.save('myDMP.npy', y)
    # print path1[-1]
    # print y[0, -1]
    print "mse"
    print mean_squared_error(path[0, :], y[0, :])
    print mean_squared_error(path[1, :], y[1, :])
    ax.plot(t, y[0], 'b--', label='DMPs')
    ax.plot(t, y[1], 'r--')
    ax.legend(loc='lower right')
    plt.show()
    # dmp.gen_force_target(path, path_v, path_a)
