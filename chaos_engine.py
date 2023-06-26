import argparse
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import json
import datetime
import time
import os
import random

# Global variable to handle pause state and program exit
pause = False
exit_program = False

def on_key_press(event):
    global pause, exit_program
    if event.key == ' ':
        pause = not pause
        if pause:
            print("Paused")
        else:
            print("Resumed")
    elif event.key == 'q':
        exit_program = True

def lorenz(x, y, z, sigma=10, rho=28, beta=8 / 3):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz

def rossler(x, y, z, alpha=0.2, beta=0.2, gamma=5.7):
    dx = -y - z
    dy = x + alpha * y
    dz = beta + z * (x - gamma)
    return dx, dy, dz

def rabinovich_fabrikant(x, y, z, alpha=0.14, gamma=0.1):
    dx = y * (z - 1 + x * x) + gamma * x
    dy = x * (3 * z + 1 - x * x) + gamma * y
    dz = -2 * z * (alpha + x * y)
    return dx, dy, dz

def logistic_map(x, r):
    return r * x * (1 - x)

def duffing_oscillator(x, y, delta=0.3, alpha=1.4, beta=-1.6, gamma=0.5, omega=1):
    dx = y
    dy = -delta * y - alpha * x - beta * x ** 3 + gamma * np.cos(omega)
    return dx, dy, 0

def get_entropy():
    # Example: Combine mouse movements and system entropy
    mouse_entropy = random.SystemRandom().random()
    system_entropy = os.urandom(8)
    combined_entropy = mouse_entropy + int.from_bytes(system_entropy, byteorder='big')
    return combined_entropy

def generate_rendering(headless=False):
    global pause, exit_program

    # Set up the simulation parameters
    dt = 0.01
    num_steps = 40000  # Increase number of steps for higher resolution

    if not headless:
        fig = plt.figure(figsize=(10, 8))  # Increase the figure size
        ax = fig.add_subplot(111, projection='3d')

        ax.view_init(elev=30, azim=120)
        ax.grid(True)  # Enable grid
        fig.canvas.mpl_connect('key_press_event', on_key_press)  # Register key press event

        for _ in range(10):  # Initial fade-out of previous renderings
            ax.cla()  # Clear previous plot
            ax.plot([], [], [], color='gray', alpha=0.2)  # Plot previous rendering
            plt.pause(0.1)

    while not exit_program:  # Add exit condition to the loop
        x_vals = np.zeros(num_steps)
        y_vals = np.zeros(num_steps)
        z_vals = np.zeros(num_steps)
        random_sequence = []

        # Generate random initial values for x, y, and z
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(0, 30)

        # Parameters for the logistic map
        logistic_r = 3.9

        # Parameters for the Duffing oscillator
        duffing_delta = 0.3
        duffing_alpha = 1.4
        duffing_beta = -1.6
        duffing_gamma = 0.5
        duffing_omega = 1

        for i in range(num_steps):
            while pause:  # Loop as long as paused
                plt.pause(0.1)

            dx_lorenz, dy_lorenz, dz_lorenz = lorenz(x, y, z)
            dx_rossler, dy_rossler, dz_rossler = rossler(x, y, z)
            dx_rabinovich, dy_rabinovich, dz_rabinovich = rabinovich_fabrikant(x, y, z)
            dx_logistic = logistic_map(x, logistic_r)
            dx_duffing, dy_duffing, _ = duffing_oscillator(x, y, duffing_delta, duffing_alpha, duffing_beta, duffing_gamma,
                                                          duffing_omega)

            dx_total = dx_lorenz + dx_rossler + dx_rabinovich + dx_logistic + dx_duffing
            dy_total = dy_lorenz + dy_rossler + dy_rabinovich + dx_duffing
            dz_total = dz_lorenz + dz_rossler + dz_rabinovich

            norm = np.sqrt(dx_total ** 2 + dy_total ** 2 + dz_total ** 2)
            dx = dx_total / norm
            dy = dy_total / norm
            dz = dz_total / norm

            x += dx * dt
            y += dy * dt
            z += dz * dt

            x_vals[i] = x
            y_vals[i] = y
            z_vals[i] = z

            # Extract a random number from the chaotic system
            random_number = x + y + z

            # Incorporate additional entropy
            additional_entropy = get_entropy()
            random_number += additional_entropy

            random_sequence.append(random_number)

        if not headless:
            ax.cla()  # Clear previous plot
            ax.grid(True)  # Enable grid
            ax.plot(x_vals, y_vals, z_vals, color='blue')  # Plot current rendering
            plt.pause(7.1)  # Adjust the interval here

        # Output random numbers to a text file
        with open('random_numbers.txt', 'w') as file:
            for number in random_sequence:
                file.write(str(number) + '\n')

        # Output random numbers to a JSON file
        with open('random_numbers.json', 'w') as file:
            json.dump(random_sequence, file)

        # Append specific settings to the log file
        with open('log.txt', 'a') as file:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"Timestamp: {timestamp}\n")
            file.write(f"Initial x: {x}, Initial y: {y}, Initial z: {z}\n")
            file.write(f"Number of steps: {num_steps}, Time step: {dt}\n")
            file.write("----------------------------------------\n")

        if not headless:
            # Respawn the visualization with new coordinates
            plt.close(fig)
            fig = plt.figure(figsize=(10, 8))  # Increase the figure size
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=30, azim=120)
            ax.grid(True)  # Enable grid
            fig.canvas.mpl_connect('key_press_event', on_key_press)  # Register key press event

def parse_arguments():
    parser = argparse.ArgumentParser(description='Chaotic 3D Rendering')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (without visualization)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    generate_rendering(headless=args.headless)
