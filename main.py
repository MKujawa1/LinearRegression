import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import random
from sklearn.linear_model import LinearRegression

def linear_curve(x,m,b):
    '''
    Retunrs linear curve. Using for curve_fit 

    Parameters
    ----------
    x : 1d array
        x axis for data.
    m : float
        m parameter for function.
    b : float
        b parameter for function.

    Returns
    -------
    TYPE
        array.

    '''
    return m*x+b

def generate_linear_data(x):
    '''
    Generate random linear curve with noise.

    Parameters
    ----------
    x : 1d array
        x axis for linear function.

    Returns
    -------
    TYPE
        array.

    '''
    m = random.uniform(-0.5, 0.5)
    b = random.uniform(-0.5, 0.5)
    noise_amp = random.uniform(0.5, 2)
    noise = np.random.randn(len(x))*noise_amp
    return (m*x+b)+noise-(noise_amp/2)

def r_squared(data, fit_data):
    '''
    Calculating r_squared value.

    Parameters
    ----------
    data : array
        raw data.
    fit_data : array
        fitted data to raw data.

    Returns
    -------
    TYPE
        float.

    '''
    correlation_array = np.corrcoef(data, fit_poly)
    correlation = correlation_array[1,0]
    return correlation**2

x = np.arange(100)
data = generate_linear_data(x)

### polyfit from numpy
params_poly = np.polyfit(x,data,1)
fit_poly = np.polyval(params_poly,x)

### curve_fit from scipy
params_cf, err = curve_fit(linear_curve, x, data)
fit_cf = linear_curve(x, *params_cf)

### LinearRegression fit
lr = LinearRegression()
lr.fit(x.reshape(-1,1),data.reshape(-1,1))
m_lr = lr.coef_[0][0]
b_lr = lr.intercept_[0]
  
### Ploting results
plt.figure(figsize = (14,5))
plt.subplot(1,3,1)
plt.title('Numpy')
plt.scatter(x,data)
plt.plot(x,fit_poly,'r',label = 'm: '+str(np.round(params_poly[0],3))+'\nb: '+str(np.round(params_poly[1],3)))
plt.legend()
plt.subplot(1,3,2)
plt.title('Curve_fit')
plt.scatter(x,data)
plt.plot(x,fit_cf,'r',label = 'm: '+str(np.round(params_cf[0],3))+'\nb: '+str(np.round(params_cf[1],3)))
plt.legend()
plt.subplot(1,3,3)
plt.title('LinearRegression')
plt.scatter(x,data)
plt.plot(x,m_lr*x+b_lr,'r',label ='m: '+str(np.round(m_lr,3))+'\nb: '+str(np.round(b_lr,3)))
plt.legend()
plt.show()

### Calculate r_squared
print('Numpy: ', r_squared(data, fit_poly))
print('Curve_fit: ', r_squared(data, fit_cf))
print('LinearRegression: ',r_squared(data, m_lr*x+b_lr))
