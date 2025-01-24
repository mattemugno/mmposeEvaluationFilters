import subprocess

PIXEL_SIZES = [i for i in range(4, 53, 4)]

########## RTMPOSE MODEL AND CONFIG PATH ##########
model_path_rtmpose_l = "models/ckpt/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth"
BASE_PATH_RTMPOSE = "models/config/pixelate/rtmpose-l_8xb256-420e_coco-256x192/"
FILE_PREFIX_RTMPOSE = "rtmpose-l_8xb256-420e_coco-256x192_"

pipeline_args_rtmpose = [
    [f"{BASE_PATH_RTMPOSE}{FILE_PREFIX_RTMPOSE}{pixel_size}.py", model_path_rtmpose_l]
    for pixel_size in PIXEL_SIZES
]

######### HRNET MODEL PATH ##########
model_path_hrnet_384x288 = "models/ckpt/td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth"
BASE_PATH_HRNET = "models/config/pixelate/td-hm_hrnet-w48_8xb32-210e_coco-384x288/"
FILE_PREFIX_HRNET = "td-hm_hrnet-w48_8xb32-210e_coco-384x288_"

pipeline_args_hrnet = [
    [f"{BASE_PATH_HRNET}{FILE_PREFIX_HRNET}{pixel_size}.py", model_path_hrnet_384x288]
    for pixel_size in PIXEL_SIZES
]

######### ViTPose MODEL PATH #########
model_path_vitpose_h = "models/ckpt/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth"
BASE_PATH_VITPOSE = "models/config/pixelate/td-hm_ViTPose-huge_8xb64-210e_coco-256x192/"
FILE_PREFIX_VITPOSE = "td-hm_ViTPose-huge_8xb64-210e_coco-256x192_"

pipeline_args_vitpose = [
    [f"{BASE_PATH_VITPOSE}{FILE_PREFIX_VITPOSE}{pixel_size}.py", model_path_vitpose_h]
    for pixel_size in PIXEL_SIZES
]

module_name = "tools/test.py"

for i, args in enumerate(pipeline_args_rtmpose, start=1):
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

        print(f"Evaluation for {model_path_vitpose_h} completed.")
    except subprocess.CalledProcessError as e:
        print(f"Errore durante l'esecuzione con argomenti {args}")
