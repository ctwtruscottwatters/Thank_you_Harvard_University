#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 00:20:46 2022

@author: Charles Thomas Wallace Truscott

Messing around after finishing a unit of Harvard.

Trying derivatives and integrals in Python

Thank you Massachusetts Institute of Technology and Harvard University


runfile('/home/truscottwatters/Desktop/charles_truscott_sympy.py', wdir='/home/truscottwatters/Desktop')
The equation is sin(0.000304617419786709*x**2) + cos(0.000304617419786709*x**2) and its derivative is -0.000609234839573417*x*sin(0.000304617419786709*x**2) + 0.000609234839573417*x*cos(0.000304617419786709*x**2)
The derivative at the point 13 is 0.0075
f'(x) at the point 13 is: 0.0075
Integrating the function -0.000609234839573417*x*sin(0.000304617419786709*x**2) + 0.000609234839573417*x*cos(0.000304617419786709*x**2) yields 0.999999999999998*sin(0.000304617419786709*x**2) + 0.999999999999998*cos(0.000304617419786709*x**2)

"""
import scipy.stats
import numpy as np
import sympy as sp
from scipy.misc import derivative
import matplotlib.pyplot as plt

def f(x):
    return np.sin((x * np.pi / 180) ** 2) + np.cos((x * np.pi / 180) ** 2)

def f_prime_of_x(x):
    return -0.000609234839573417 * x * np.sin(0.000304617419786709 * x ** 2) + 0.000609234839573417 * x * np.cos(0.000304617419786709 * x ** 2)
def calculus_imperative():
    plt.figure(0, dpi=300, figsize=[16, 8])
    plt.title("Charles Thomas Wallace Truscott")
    x = sp.symbols('x')
    equation = sp.sin((x * np.pi / 180) ** 2) + sp.cos((x * np.pi / 180) ** 2)
    derivative_symbolic = equation.diff(x)
    point = 13
    print("The equation is {} and its derivative is {}".format(equation, derivative_symbolic))
    print("The derivative at the point {} is {}".format(str(point), round(derivative(f, x0=point), 4)))
    print("f'(x) at the point 13 is: {}".format(round(f_prime_of_x(point), 4)))
    integral = -0.000609234839573417 * x * sp.sin(0.000304617419786709 * x ** 2) + 0.000609234839573417 * x * sp.cos(0.000304617419786709 * x ** 2)
    print("Integrating the function {} yields {}".format(integral, sp.integrate(integral, x)))
    r = []
    L = []
    R = []
    for q in range(0, 360, 1):
        r.append(q)
        L.append(np.sin((q * np.pi / 180) ** 2) + np.cos((q * np.pi / 180) ** 2))
        R.append(-0.000609234839573417 * q * np.sin(0.000304617419786709 * q ** 2) + 0.000609234839573417 * q * np.cos(0.000304617419786709 * q ** 2))
    plt.plot(r, L, color='crimson', ms=12, label="f(x)")
    plt.plot(r, R, color='grey', ms=12, label="f'(x)")
    plt.legend()
    plt.savefig('derivatives_sympy.pdf', dpi=300, papertype='a4')
if __name__ == "__main__": calculus_imperative()