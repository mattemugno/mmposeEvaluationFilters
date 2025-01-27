import json
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('qtagg')
from mpl_toolkits.mplot3d import Axes3D

data_root = "tools/json_results/rtmpose-l/blur/results"

def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def plot_3d(data_files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for file in data_files:
        data = load_data(file)

        keypoints = list(data.keys())[10:]
        errors = [data[k] for k in keypoints]
        intensity = float(file.split("_")[-1].split(".")[0].replace("blur", "").split("x")[0])

        z = np.full_like(errors, intensity, dtype=float)
        ax.scatter(np.arange(len(keypoints)), errors, z, label=f"Intensity: {intensity}x{intensity}", marker='o')

    ax.set_xlabel('Keypoint')
    ax.set_ylabel('Error')
    ax.set_zlabel('Kernel Size')
    ax.set_title('3D Plot of Keypoints, Error, and Filter Intensity')
    plt.xticks(rotation=90)
    plt.show()


def plot_heatmap(data_files):
    data = []
    intensities = []

    for file in data_files:
        json_data = load_data(file)
        intensity = float(file.split("_")[-1].split(".")[0].replace("blur", "").split("x")[0])
        intensities.append(intensity)

        for keypoint, error in json_data.items():
            if keypoint not in ['AP', 'AP. 5', 'AP. 75', 'AP(M)', 'AP(L)', 'AR', 'AR. 5', 'AR. 75', 'AR(M)', 'AR(L)']:
                data.append({'Keypoint': keypoint, 'Intensity': intensity, 'Error': error})

    df = pd.DataFrame(data)
    df_pivot = df.pivot(index='Keypoint', columns='Intensity', values='Error')

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Mean Error'})
    plt.title("Mean Error per Keypoint and Filter Intensity")
    plt.xlabel("Filter Intensity")
    plt.ylabel("Keypoint")
    plt.show()


def plot_3d_surface(data_files):
    keypoints = set()
    intensities = []
    errors = []

    for file in data_files:
        data = load_data(file)
        intensity = float(file.split("_")[-1].split(".")[0].replace("blur", "").split("x")[0])
        intensities.append(intensity)

        keypoint_errors = {k: v for k, v in data.items() if
                           k not in ['AP', 'AP. 5', 'AP. 75', 'AP(M)', 'AP(L)', 'AR', 'AR. 5', 'AR. 75', 'AR(M)',
                                     'AR(L)']}

        keypoints.update(keypoint_errors.keys())
        errors.append([keypoint_errors.get(k, 0) for k in sorted(keypoints)])

    keypoints = sorted(keypoints)
    intensities = sorted(intensities)
    errors = np.array(errors)

    x = np.arange(len(keypoints))
    y = np.array(intensities)
    X, Y = np.meshgrid(x, y)

    Z = errors

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

    ax.set_xticks(np.arange(len(keypoints)))
    ax.set_xticklabels(keypoints, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Keypoint')
    ax.set_ylabel('Mean Error')
    ax.set_zlabel('Filter Intensity')
    ax.set_title('3D Heatmap: Mean Error by Keypoint and Filter Intensity')

    # Barra dei colori
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Mean Error')

    plt.show()


data_files = [os.path.join(data_root, file) for file in os.listdir(data_root)]

plot_3d(data_files)
#plot_heatmap(data_files)
#plot_3d_surface(data_files)