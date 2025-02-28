import subprocess

KERNEL_SIZES_DONE = ["5x5", "17x17", "29x29", "41x41", "53x53", "65x65", "77x77", "89x89", "101x101"]
KERNEL_SIZES = [f"{i}x{i}" for i in range(5, 102, 4) if f"{i}x{i}" not in KERNEL_SIZES_DONE]

########## RTMPOSE MODEL AND CONFIG PATH ##########
model_path_rtmpose_l = "models/ckpt/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
BASE_PATH_RTMPOSE_BLUR = "models/config/blur/rtmpose-l_8xb256-420e_coco-256x192/"
FILE_PREFIX_RTMPOSE = "rtmpose-l_8xb256-420e_coco-256x192_"

pipeline_args_rtmpose = [
    [f"{BASE_PATH_RTMPOSE_BLUR}{FILE_PREFIX_RTMPOSE}{kernel_size}.py", model_path_rtmpose_l]
    for kernel_size in KERNEL_SIZES
]

######### HRNET MODEL PATH ##########
model_path_hrnet_384x288 = "models/ckpt/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth"
BASE_PATH_HRNET_BLUR = "models/config/blur/td-hm_hrnet-w48_8xb32-210e_coco-384x288/"
FILE_PREFIX_HRNET = "td-hm_hrnet-w48_8xb32-210e_coco-384x288_"

pipeline_args_hrnet = [
    [f"{BASE_PATH_HRNET_BLUR}{FILE_PREFIX_HRNET}{kernel_size}.py", model_path_hrnet_384x288]
    for kernel_size in KERNEL_SIZES
]

######### ViTPose MODEL PATH #########
model_path_vitpose_b = "models/ckpt/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"
BASE_PATH_VITPOSE_BLUR = "models/config/blur/td-hm_ViTPose-base_8xb64-210e_coco-256x192/"
FILE_PREFIX_VITPOSE = "td-hm_ViTPose-base_8xb64-210e_coco-256x192_"

pipeline_args_vitpose = [
    [f"{BASE_PATH_VITPOSE_BLUR}{FILE_PREFIX_VITPOSE}{kernel_size}.py", model_path_vitpose_b]
    for kernel_size in KERNEL_SIZES[-2:]
]

########################################
############ BLUR EVALUATION ###########
########################################

module_name = "tools/test.py"

'''for i, args in enumerate(pipeline_args_rtmpose, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_rtmpose_l} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")

for i, args in enumerate(pipeline_args_vitpose, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_vitpose_b} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")'''

'''for i, args in enumerate(pipeline_args_hrnet, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_hrnet_384x288} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")'''

########################################
######### PIXELATE EVALUATION ##########
########################################

import subprocess

PIXEL_SIZES = [4, 8, 12, 16]

########## RTMPOSE MODEL AND CONFIG PATH ##########
BASE_PATH_RTMPOSE_PIXELATE = "models/config/pixelate/rtmpose-l_8xb256-420e_coco-256x192/"

pipeline_args_rtmpose = [
    [f"{BASE_PATH_RTMPOSE_PIXELATE}{FILE_PREFIX_RTMPOSE}{pixel_size}.py", model_path_rtmpose_l]
    for pixel_size in PIXEL_SIZES
]

######### HRNET MODEL PATH ##########
BASE_PATH_HRNET_PIXELATE = "models/config/pixelate/td-hm_hrnet-w48_8xb32-210e_coco-384x288/"

pipeline_args_hrnet = [
    [f"{BASE_PATH_HRNET_PIXELATE}{FILE_PREFIX_HRNET}{pixel_size}.py", model_path_hrnet_384x288]
    for pixel_size in PIXEL_SIZES
]

######### ViTPose MODEL PATH #########
BASE_PATH_VITPOSE_PIXELATE = "models/config/pixelate/td-hm_ViTPose-base_8xb64-210e_coco-256x192/"

pipeline_args_vitpose = [
    [f"{BASE_PATH_VITPOSE_PIXELATE}{FILE_PREFIX_VITPOSE}{pixel_size}.py", model_path_vitpose_b]
    for pixel_size in PIXEL_SIZES
]

module_name = "tools/test.py"

'''for i, args in enumerate(pipeline_args_rtmpose, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_rtmpose_l} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")

for i, args in enumerate(pipeline_args_vitpose, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_vitpose_b} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")'''


for i, args in enumerate(pipeline_args_hrnet, start=1):
    intensity = args[0].split("_")[-1].split(".")[0]

    try:
        print(f"Args: {args}.")

        result = subprocess.run(
            ["python", module_name] + args,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )

        print(f"Evaluation for {model_path_hrnet_384x288} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")

