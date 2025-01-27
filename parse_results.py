import json
import os

data_root = "tools/json_results/rtmpose-l/pixelate/results"

def extract_kernel(filename):
    try:
        intensity = filename.split('_')[2].replace('.json', '')
        kernel_size = int(intensity.split('x')[0])
        return kernel_size
    except (IndexError, ValueError):
        return float('inf')


def json_to_latex_table(json_files, output_file):
    latex_table = (
        r"\begin{table}[h!]\n"
        r"\centering\n"
        r"\begin{tabular}{|c|c|" + "c|" * 10 + r"}\hline"
    )

    headers = list(json.load(open(os.path.join(data_root, json_files[0]))).keys())[:10]
    latex_table += "Model & Intensity & " + " & ".join(headers) + r" \\\\ \hline"

    for json_file in json_files:
        filename = os.path.basename(json_file)
        model, intensity = filename.split('_')[1], filename.split('_')[2].split('.')[0]

        with open(os.path.join(data_root, filename), 'r') as f:
            data = json.load(f)

        values = [round(x, 2) for x in list(data.values())[:10]]
        latex_table += f"{model} & {intensity} & " + " & ".join(map(str, values)) + r" \\\\ \hline"

    latex_table += (
        r"\end{tabular}"
        r"\caption{Tabella generata dai dati JSON.}"
        r"\label{tab:json_table}"
        r"\end{table}"
    )

    with open(output_file, 'w') as f:
        f.write(latex_table)

    print(f"Tabella LaTeX salvata in {output_file}")


json_files = sorted(os.listdir(data_root), key=extract_kernel)
output_file = "table.tex"
json_to_latex_table(json_files, output_file)
