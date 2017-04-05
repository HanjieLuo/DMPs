# -*- coding: utf-8 -*- 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
from lwpr import LWPR
from sklearn.metrics import mean_squared_error
import time


class Canonical(object):
    def __init__(self, dt, kernel_num, ax):
        self.dt = dt
        self.ax = ax
        self.kernel_num = kernel_num
        self.x = 1.0

    def run(self, t, tau=None):
        if tau is None:
            # 如果tau为None, t表示为归一化后的时间
            return np.exp(- self.ax * t)
        else:
            # 否则t表示具体的时间
            return np.exp(- self.ax * t / tau)


class DMP(object):
    def __init__(self, system_dt=0.001):
        self.kernel_num = 20
        self.az = 48  # 48
        self.bz = self.az / 4
        # self.ax = self.az / 3
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

        self.f_target = self.gen_force_target(path, path_v, path_a)
        self.init_LWPR()
        self.lwpr_learn()

    def init_LWPR(self):
        self.lwpr = []
        for i in range(self.joints_num):
            lwpr = LWPR(1, 1)
            lwpr.init_D = 10000 * np.eye(1)
            lwpr.update_D = True
            lwpr.init_alpha = 10 * np.eye(1)
            lwpr.meta = False
            lwpr.penalty = 0.000000001
            # lwpr.w_gen = 0.2

            # lwpr.init_D = 200 * np.eye(1)
            # lwpr.update_D = True
            # lwpr.init_alpha = 0.1 * np.eye(1)
            # lwpr.meta = False
            # lwpr.penalty = 0.005
            # lwpr.w_gen = 0.2
            # lwpr.w_prune = 0.8

            # double   w_gen=0.2;
            # double   w_prune=0.8;
            # bool     update_D=true;
            # double   init_alpha=0.1;
            # double   penalty=0.005;
            # VectorXd init_D=VectorXd::Constant(input_dim,200);
            self.lwpr.append(lwpr)

    def lwpr_learn(self):
        t = np.linspace(0.0, self.tau, self.steps)
        x = (self.canonical.run(t, self.tau)).reshape((self.steps, 1))
        for j in range(self.joints_num):
            force = self.f_target[j, :].reshape((self.steps, 1))
            x2 = x * (self.goals[j] - self.y0[j])
            for i in range(self.steps):
                # print x[i], x2[i], force[i]
                self.lwpr[j].update(x2[i], force[i])
                # self.lwpr[j].update(x[i], force[i])
            print "For jonit", j, "use ", self.lwpr[j].num_rfs, "kernels"

        # f = np.zeros((self.joints_num, self.steps))
        # for i in range(self.joints_num):
        #     x2 = x * (self.goals[i] - self.y0[i])
        #     # print x2
        #     for j in range(self.steps):
        #         f[i, j], _ = self.lwpr[i].predict_conf(x2[j])
            # x2 = x2.reshape((self.steps, 1))
            # plt.plot(x2, self.f_target[i, :], 'r--')
            # plt.plot(x2, f[i, :], 'b')
            # plt.show()

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
        f = np.zeros(self.joints_num)
        for i in range(self.joints_num):
            x2 = x * (self.goals[i] - self.y0[i])
            # x2 = x
            # print "p,", self.goals[i] - self.y0[i]
            f[i], _ = self.lwpr[i].predict_conf(np.array([x2]))
            # f[i], _ = self.lwpr[i].predict_conf(np.array([x]))
        return f

    def run(self):
        if (self.t > self.tau):
            return

        while True:
            self.y += self.y_v * self.internal_dt
            self.y_v += self.y_a * self.internal_dt

            x = self.canonical.run(self.t, self.tau)
            f = self.gen_force(x)
            self.y_a = (self.az * (self.bz * (self.goals - self.y) - self.tau * self.y_v) + f) / (self.tau ** 2)

            self.t += self.internal_dt
            self.cur_step += 1

            if self.cur_step % self.steps_between_measurement == 0:
                break
        return self.y, self.y_v, self.y_a


if __name__ == "__main__":
    mytime = 8
    dt = 0.01
    path1 = np.cos(np.arange(0, mytime, dt) * 1)
    path2 = np.sin(np.arange(0, mytime, dt) * 1)
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
    dmp.init(path, path_v, path_a, mytime, True)
    print("--- %s seconds ---" % (time.time() - start_time))

    t = np.arange(0, mytime, dt)

    fig, ax = plt.subplots()
    ax.plot(t, path[0], 'b-', label='target')
    ax.plot(t, path[1], 'r-')

    # print dmp.y0
    # print dmp.goals
    dmp.tau = mytime = 4
    t = np.arange(0, mytime, dt)
    # dmp.goals[1] = 0.6

    y = np.zeros((2, mytime/dt))
    y_v = np.empty_like(y)
    y_a = np.empty_like(y)

    for i in range(int(mytime / dt)):
        y[:, i], y_v[:, i], y_a[:, i] = dmp.run()

    # y2 = np.load('myDMP.npy')
    # print "mse"
    # print mean_squared_error(path[0, :], y[0, :])
    # print mean_squared_error(path[1, :], y[1, :])

    ax.plot(t, y[0], 'b--', label='DMPs with LWPR')
    ax.plot(t, y[1], 'r--')
    # ax.plot(t, y2[0], 'b:', label='DMPs with LWR')
    # ax.plot(t, y2[1], 'r:')
    ax.legend(loc='lower right')
    plt.show()

