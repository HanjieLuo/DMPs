import numpy as np
import matplotlib.pyplot as plt
def spring_damper(x0, g, tau, dt, alpha, beta, observe_for):
    X = [x0]
    xd = 0.0
    xdd = 0.0
    t = 0.0
    while t < observe_for:
        X.append(X[-1] + xd * dt)
        x = X[-1]
        xd += xdd * dt
        xdd = alpha / (tau ** 2) * (beta * (g - x) - tau * xd)
        t += dt
    return X

x0, g = 0.0, 1.0
tau = 1.0
observe_for = 2.0 * tau
dt = 0.01

plt.figure(figsize=(10, 5))
plt.xlabel("Time")
plt.ylabel("Position")
plt.xlim((0.0, observe_for))
diff = g - x0
plt.ylim((x0 - 0.1 * diff, g + 0.5 * diff))
for alpha, beta in [(25.0, 6.25), (25.0, 1.5), (25.0, 25.0)]:
    X = spring_damper(x0, g, tau, dt, alpha, beta, observe_for)
    plt.plot(np.arange(0.0, observe_for + dt, dt), X, lw=3,
             label="$\\alpha = %g,\\quad \\beta = %g$" % (alpha, beta))
plt.scatter([0.0, tau], [x0, g], marker="*", s=500, label="$x_0, g$")
plt.legend(loc="lower right")
plt.show()