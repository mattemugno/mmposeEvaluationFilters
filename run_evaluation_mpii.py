import os
import subprocess

PIXEL_SIZES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

######### HRNET MODEL PATH ##########
'''model_path_hrnet_384x288 = "models/ckpt/hrnet_w48_mpii_256x256-92cab7bd_20200812.pth"
FILE_PREFIX_HRNET = "td-hm_hrnet-w48_8xb64-210e_mpii-256x256_"
BASE_PATH_HRNET_PIXELATE = "models/config/pixelate/td-hm_hrnet-w48_8xb64-210e_mpii-256x256"

module_name = "tools/test.py"
dump_result_folder = "tools/json_results/hrnet-mpii/pixelate/dump/"
metrics_folder = "tools/json_results/hrnet-mpii/pixelate/metrics/"

for pixel_size in PIXEL_SIZES:
    config_path = f"{BASE_PATH_HRNET_PIXELATE}/{FILE_PREFIX_HRNET}{pixel_size}.py"
    out_file = f"{metrics_folder}pixelate_{pixel_size}.json"
    dump_file = f"{dump_result_folder}predictions_pixelate_{pixel_size}.pkl"

    if not os.path.exists(config_path):
        print(f"Errore: Configuration file {config_path} does not exists!")
        continue

    args = ["python", module_name, config_path, model_path_hrnet_384x288, "--out", out_file, "--dump", dump_file]

    try:
        print(f"Running: {' '.join(args)}")

        result = subprocess.run(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for pixel size {pixel_size} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con pixel size {pixel_size}.")
        print(f"Output errore: {e.stderr}")
'''

######### RTMPOSE PIXELATE MODEL PATH ##########

'''model_rtmpose = "models/ckpt/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth"
FILE_PREFIX_RTMPOSE = "rtmpose-m_8xb64-210e_mpii-256x256_"
BASE_PATH_RTMPOSE_PIXELATE = "models/config/pixelate/rtmpose-m_8xb64-210e_mpii-256x256"

module_name = "tools/test.py"
dump_result_folder = "tools/json_results/rtmpose-mpii/pixelate/dump/"
metrics_folder = "tools/json_results/rtmpose-mpii/pixelate/metrics/"

for pixel_size in PIXEL_SIZES:
    config_path = f"{BASE_PATH_RTMPOSE_PIXELATE}/{FILE_PREFIX_RTMPOSE}{pixel_size}.py"
    out_file = f"{metrics_folder}pixelate_{pixel_size}.json"
    dump_file = f"{dump_result_folder}predictions_pixelate_{pixel_size}.pkl"

    if not os.path.exists(config_path):
        print(f"Errore: Configuration file {config_path} does not exists!")
        continue

    args = ["python", module_name, config_path, model_rtmpose, "--out", out_file, "--dump", dump_file]

    try:
        print(f"Running: {' '.join(args)}")

        result = subprocess.run(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for pixel size {pixel_size} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con pixel size {pixel_size}.")
        print(f"Output errore: {e.stderr}")'''


######### RTMPOSE BLUR MODEL PATH ##########

'''model_rtmpose = "models/ckpt/rtmpose-m_simcc-mpii_pt-aic-coco_210e-256x256-ec4dbec8_20230206.pth"
FILE_PREFIX_RTMPOSE = "rtmpose-m_8xb64-210e_mpii-256x256_"
BASE_PATH_RTMPOSE_BLUR = "models/config/blur/rtmpose-m_8xb64-210e_mpii-256x256"

module_name = "tools/test.py"
dump_result_folder = "tools/json_results/rtmpose-mpii/blur/dump/"
metrics_folder = "tools/json_results/rtmpose-mpii/blur/metrics/"

KERNEL_SIZES = [f"{i}x{i}" for i in range(5, 102, 2) if f"{i}x{i}"]

for kernel_size in KERNEL_SIZES:
    config_path = f"{BASE_PATH_RTMPOSE_BLUR}/{FILE_PREFIX_RTMPOSE}{kernel_size}.py"
    out_file = f"{metrics_folder}blur_{kernel_size}.json"
    dump_file = f"{dump_result_folder}predictions_blur_{kernel_size}.pkl"

    if not os.path.exists(config_path):
        print(f"Errore: Configuration file {config_path} does not exists!")
        continue

    args = ["python", module_name, config_path, model_rtmpose, "--out", out_file, "--dump", dump_file]

    try:
        print(f"Running: {' '.join(args)}")

        result = subprocess.run(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for kernel size {kernel_size} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con kernel size {kernel_size}.")
        print(f"Output errore: {e.stderr}")'''


######### VITPOSE BLUR MODEL PATH ##########

model_vitpose = "models/ckpt/vitpose_base_coco_aic_mpii.pth"
FILE_PREFIX_VITPOSE = "td-hm_ViTPose-base_8xb64-210e_coco-256x192_"
BASE_PATH_VITPOSE_BLUR = "models/config/pixelate/vitpose-mpii"

module_name = "tools/test.py"
dump_result_folder = "tools/json_results/vitpose-mpii/blur/dump/"
metrics_folder = "tools/json_results/vitpose-mpii/blur/metrics/"

KERNEL_SIZES = [f"{i}x{i}" for i in range(5, 102, 2) if f"{i}x{i}"]

for kernel_size in KERNEL_SIZES:
    config_path = f"{BASE_PATH_VITPOSE_BLUR}/{FILE_PREFIX_VITPOSE}{kernel_size}.py"
    out_file = f"{metrics_folder}blur_{kernel_size}.json"
    dump_file = f"{dump_result_folder}predictions_blur_{kernel_size}.pkl"

    if not os.path.exists(config_path):
        print(f"Errore: Configuration file {config_path} does not exists!")
        continue

    args = ["python", module_name, config_path, model_vitpose, "--out", out_file, "--dump", dump_file]

    try:
        print(f"Running: {' '.join(args)}")

        result = subprocess.run(
            args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for kernel size {kernel_size} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con kernel size {kernel_size}.")
        print(f"Output errore: {e.stderr}")