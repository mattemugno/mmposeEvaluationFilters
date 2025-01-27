import re
import os

def modify_kernel_size(config_path, new_kernel_size, output_path):
    with open(config_path, 'r') as file:
        config_content = file.read()

    updated_content = re.sub(
        r"(dict\(type='ApplyGaussianBlur',\s*kernel_size=)\(\d+,\s*\d+\)",
        f"\\1{tuple(new_kernel_size)}",
        config_content
    )

    updated_content = re.sub(
        r"(outfile_prefix='[^']*blur/format_only/)",
        lambda m: f"{m.group(1)}{new_kernel_size[0]}x{new_kernel_size[1]}",
        updated_content
    )

    with open(output_path, 'w') as file:
        file.write(updated_content)

    print(f"File salvato con successo in: {output_path}")

configs_folder = f'models/config/blur/td-hm_ViTPose-base_8xb64-210e_coco-256x192'
config = 'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'

kernel_value = [5, 5]

for _ in range(30):
    output_path = f'{configs_folder}/td-hm_ViTPose-base_8xb64-210e_coco-256x192_{kernel_value[0]}x{kernel_value[1]}.py'
    modify_kernel_size(os.path.join(configs_folder, config), kernel_value, output_path)
    kernel_value[0] += 4
    kernel_value[1] += 4