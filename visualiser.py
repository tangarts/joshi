#!/usr/bin/env python3

#import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-dark')


# plot simulated asset paths from monte carlo
def price_path_plot(data, simulations):
    # data : data to be plotted
    # simulations : number of simulations of the data to be plotted

    plt.figure(figsize=(15, 10))
    plt.plot(data[:, :simulations])
    plt.grid(True)
    plt.title('{} Asset Price Simulation'.format(simulations))
    plt.xlabel('time step')
    plt.ylabel('price level')
    plt.show()

# asset price histogram
def price_hist(data):
    plt.figure(figsize=(13, 7))
    plt.hist(data[-1], bins=70)
    plt.grid(True)
    plt.title('Asset price distribution')
    plt.xlabel('Asset prices')
    plt.ylabel('Frequency')
    plt.show()

