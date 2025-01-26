import re
import os

def modify_kernel_size(config_path, new_pixel_size, output_path):
    with open(config_path, 'r') as file:
        config_content = file.read()

    updated_content = re.sub(
        r"(dict\(type='ApplyPixelation',\s*pixel_size=)\d+",
        f"\\g<1>{new_pixel_size}",
        config_content
    )

    updated_content = re.sub(
        r"(outfile_prefix='[^']*/pixelate/format_only/)(\d+)?'",
        lambda m: f"{m.group(1)}{new_pixel_size}'",
        updated_content
    )

    with open(output_path, 'w') as file:
        file.write(updated_content)

    print(f"File salvato con successo in: {output_path}")

configs_folder = 'models/config/pixelate/td-hm_ViTPose-huge_8xb64-210e_coco-256x192'
pixel_size = 4

for config in os.listdir(configs_folder):
    output_path = f'{configs_folder}/td-hm_ViTPose-huge_8xb64-210e_coco-256x192_{pixel_size}.py'
    modify_kernel_size(os.path.join(configs_folder, config), pixel_size, output_path)
    pixel_size += 4