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

    '''updated_content = re.sub(
        r"(outfile_prefix='[^']*/pixelate/format_only/)(\d+)?'",
        lambda m: f"{m.group(1)}{new_pixel_size}'",
        updated_content
    )'''

    updated_content = re.sub(
        r'(output_prefix\s*=\s*"[^"]*/pixelate/format_only/)(?:"|\d*)',
        lambda m: f'{m.group(1)}{new_pixel_size}"',
        updated_content
    )

    with open(output_path, 'w') as file:
        file.write(updated_content)

    print(f"File salvato con successo in: {output_path}")

configs_folder = 'models/config/pixelate/rtmpose-m_8xb64-210e_mpii-256x256'
config = 'rtmpose-m_8xb64-210e_mpii-256x256.py'
pixel_size = 2

for _ in range(20):
    output_path = f'{configs_folder}/rtmpose-m_8xb64-210e_mpii-256x256_{pixel_size}.py'
    modify_kernel_size(os.path.join(configs_folder, config), pixel_size, output_path)
    pixel_size += 1