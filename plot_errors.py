import json
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

matplotlib.use('qtagg')
from mpl_toolkits.mplot3d import Axes3D


def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

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

    surf = ax.plot_surface(X, Z, Y, cmap='viridis', edgecolor='k', alpha=0.8)

    ax.set_xticks(np.arange(len(keypoints)))
    ax.set_xticklabels(keypoints, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Keypoint')
    ax.set_ylabel('Filter Intensity')
    ax.set_zlabel('Mean Error')
    ax.set_title('3D Heatmap: Mean Error by Keypoint and Filter Intensity')

    # Barra dei colori
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Mean Error')

    plt.show()

def plot_3d(data_files):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), subplot_kw={'projection': '3d'})

    fig.patch.set_facecolor('white')

    all_x = []
    all_y = []
    all_z = []

    keypoint_names = ["Nose", "LEye", "REye", "LEar", "REar",
                      "LShoulder", "RShoulder", "LElbow", "RElbow",
                      "LWrist", "RWrist", "LHip", "RHip",
                      "LKnee", "RKnee", "LAnkle", "RAnkle"]

    for file in data_files:
        data = load_data(file)
        keypoints = list(data.keys())[10:]
        errors = np.array([data[k] for k in keypoints])

        intensity = int(file.split("_")[-1].split(".")[0].split("x")[0])

        x_indices = np.arange(len(keypoints))
        z = np.full_like(errors, intensity, dtype=float)

        all_x.extend(x_indices)
        all_y.extend(z)
        all_z.extend(errors)

    xi = np.linspace(min(all_x), max(all_x), 50)
    yi = np.linspace(min(all_y), max(all_y), 50)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((all_x, all_y), all_z, (X, Y), method='cubic')

    for i, ax in enumerate(axes):
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

        ax.set_facecolor('white')

        # Rimuove il background grigio dalle 'panne' degli assi 3D
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Imposta il colore di sfondo dei bordi degli assi
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        # ax.set_xticks(np.arange(len(keypoints)))
        # ax.set_xticklabels(np.arange(0, 17), ha='right', fontsize=10)
        ax.set_xticks(np.arange(len(keypoint_names)))

        if i == 0:
            ax.set_xticklabels(keypoint_names, ha='left', fontsize=9)
        else:
            ax.set_xticklabels(keypoint_names, ha='right', fontsize=9, rotation=90)

        # ax.set_xticklabels(keypoint_names, ha='right', fontsize=9, rotation=90)

        ax.tick_params(axis='x', labelsize=9)
        # ax.set_ylabel('Kernel Size', fontsize=12)
        ax.set_ylabel('Pixel Size', fontsize=12)
        # ax.set_xlabel('Keypoint', fontsize=12)
        ax.set_zlabel('Error', fontsize=12)

        ax.set_facecolor('#f4f4f4')
        ax.grid(True, linestyle=':', color='white', linewidth=0.5)

    fig.suptitle("RTMPose-L Blur", fontsize=16)

    axes[0].view_init(elev=20, azim=20)
    axes[1].view_init(elev=30, azim=60)
    axes[2].view_init(elev=40, azim=90)

    # Aggiunta della legenda con gli indici dei keypoints
    #legend_text = "\n".join([f"{i}: {name}" for i, name in enumerate(keypoint_names)])
    #fig.text(0.82, 0.5, legend_text, fontsize=10, verticalalignment='center',
    #         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    fig.savefig("RTMPose-L Blur.png", dpi=300, bbox_inches='tight')

    plt.show()



models = ["hrnet-384x288", "rtmpose-l", "vitpose-b"]
filter = ["blur", "pixelate"]

data_root = "tools/json_results/rtmpose-l/blur/results"

data_files = [os.path.join(data_root, file) for file in os.listdir(data_root)]

plot_3d(data_files)
#plot_heatmap(data_files)
#plot_3d_surface(data_files)