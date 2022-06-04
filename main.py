import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn.linear_model import LinearRegression

def linear_curve(x,m,b):
    return m*x+b

def generate_linear_data(x):
    m = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)
    noise_amp = random.uniform(0.5, 2)
    noise = np.random.randn(len(x))*noise_amp
    return (m*x+b)+noise-(noise_amp/2)

x = np.arange(100)
data = generate_linear_data(x)

### polyfit from numpy
params_poly = np.polyfit(x,data,1)
fit_poly = np.polyval(params_poly,x)

### curve_fit from scipy
params_cf, err = curve_fit(linear_curve, x, data)
fit_cf = linear_curve(x, *params_cf)
    
plt.figure(figsize = (14,5))
plt.subplot(1,3,1)
plt.title('Numpy')
plt.scatter(x,data)
plt.plot(fit_poly,'r',label = str(np.round(params_poly[0],3))+'*x + '+str(np.round(params_poly[0],3)))
plt.legend()
plt.subplot(1,3,2)
plt.title('Curve_fit')
plt.scatter(x,data)
plt.plot(fit_cf,'r',label = str(np.round(params_cf[0],3))+'*x + '+str(np.round(params_cf[0],3)))
plt.legend()
plt.subplot(1,3,3)
plt.title('Numpy')
plt.scatter(x,data)
plt.plot(fit_poly,'r',label = str(np.round(params_poly[0],3))+'*x + '+str(np.round(params_poly[0],3)))
plt.legend()

