import numpy as np
from matplotlib import pyplot as plt
from lwpr import LWPR

time = 2
dt = 0.01
Ntr = int(time / dt)
# t = np.arange(0, time, dt)
# Xtr = np.exp(- abs(np.log(0.0000001)) * t / time)
# print Xtr
ax = 2
t = np.arange(0, time, dt)
Ytr = np.sin(t)
Xtr = np.exp(- ax * t / time) * (Ytr[-1] - Ytr[0])
# print Xtr
# print Ytr

Xtr = Xtr.reshape((Ntr, 1))
Ytr = Ytr.reshape((Ntr, 1))

# initialize the LWPR model
model = LWPR(1, 1)
model.init_D = 10000 * np.eye(1)
model.update_D = True
model.init_alpha = 40 * np.eye(1)
model.meta = True
model.penalty = 0.00000000001
print model

# train the model
# for k in range(20):
#     ind = np.random.permutation(Ntr)
#     mse = 0

#     for i in range(Ntr):
#         yp = model.update(Xtr[ind[i]], Ytr[ind[i]])
#         mse = mse + (Ytr[ind[i], :] - yp)**2

#     nMSE = mse / Ntr / np.var(Ytr)
#     print "#Data: %5i  #RFs: %3i  nMSE=%5.3f" % (model.n_data, model.num_rfs, nMSE)

for i in range(Ntr):
    model.update(Xtr[i], Ytr[i])
print model.num_rfs


# test the model with unseen data
Ntest = 5000
Ttest = np.linspace(0, time, Ntest)

Xtest = np.exp(- ax * Ttest / time)
Ytest = np.zeros((Ntest, 1))
Conf = np.zeros((Ntest, 1))

for k in range(Ntest):
    Ytest[k, :], Conf[k, :] = model.predict_conf(np.array([Xtest[k]* (Ytr[-1] - Ytr[0])]))

plt.plot(t, Ytr, 'r.')
# plt.plot(t, Xtr, 'b.')
# plt.plot(Xtr, Ytr, 'y.')
plt.plot(Ttest, Ytest, 'g-')
# plt.plot(Xtest, Ytest + Conf, 'c-', linewidth=2)
# plt.plot(Xtest, Ytest - Conf, 'c-', linewidth=2)

plt.show()
