from noisyopt import minimizeCompass
import numpy as np
import matplotlib.pyplot as plt


domain = [0., 1.]  # the domain is [[0, 1]]

# target_0 = lambda x : x * (1-x) * (4 - np.sqrt(np.abs(np.sin(60 * x))))
# target = lambda x : target_0(x) + target_0(x-0.01)
target = lambda x : (x-0.)**2 + 1

var = 0.1  # noise variance

def obj(x):
    return target(x) + var * np.random.uniform(x.shape)


bound = np.array(domain).reshape(1,2)
x0 = np.array([0.5])
res = minimizeCompass(obj, bounds=bound, x0=x0, deltatol=10*var, paired=False)
print(res)
res2 = minimizeCompass(obj, bounds=bound, x0=x0, errorcontrol = True, paired=False)
print(res2)


fig = plt.figure()
plt.subplot(1,3,1)
ax= plt.gca()
x = np.linspace(domain[0], domain[1], 1000)
z = target(x) 

ax.plot(x, z+var, '-r', alpha = 0.1)
ax.plot(x, z, alpha=0.9)
ax.plot(x, z-var, '-r', alpha = 0.1)
plt.axvline(res.x, color = 'red')

plt.show()