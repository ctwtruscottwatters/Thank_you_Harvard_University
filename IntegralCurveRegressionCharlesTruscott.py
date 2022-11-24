import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import scipy.integrate as si
import random
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def f(x, y):
	return y - np.sin(x) + np.cos(x) + x
	
def main():
	""" Not as efficient as Runge Kutta or Euler's Method but, this is Polynomial Interpolation AI trained to solve the integral curves of a Trigonometric Ordinary Differential Equation. Charles Truscott """
	
	plt.figure(0, dpi=120, figsize=[10, 10])
	plt.title("Charles Truscott Watters on a Caterpillar S60 Rugged Phone, training Artificial Intelligence")
	x = np.linspace(-2, 2, 40)
	y = np.linspace(-2, 2, 40)
	x, y = np.meshgrid(x, y)
	dy = -np.cos(x) + np.sin(x)
	dx = np.ones(dy.shape)
	x2 = np.linspace(-2 , 2, 40)
	y2 = np.linspace(-2, 2, 40)
	x2, y2 = np.meshgrid(x2, y2)
	plt.quiver(x, y, dy, dx, color='black')
	plt.contour(x2, y2, f(x2, y2), cmap='rainbow')
	x3, y3 = np.linspace(-2, 2, 40), np.linspace(-2, 2, 40)
#	x3 = x3[:,np.newaxis]
	x3, y3 = np.meshgrid(x3, y3)
	y4 = f(x3, y3)
	colors = ["viridis", "terrain"]
	for count, degree in enumerate([3, 5]):
		model = make_pipeline(PolynomialFeatures(degree), Ridge())
		model.fit(x3, y4)
		y_plot = model.predict(x3)
		y_plot = y3 + y_plot - x3
		plt.contour(x3, y3, y_plot, cmap=colors[count])
	plt.show()
if __name__ == "__main__": main()