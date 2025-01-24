import json

def process_json(data):
    updated_data = []
    for item in data:
        item.pop("bbox", None)
        item["keypoints"] = [round(kp) if isinstance(kp, float) else kp for kp in item["keypoints"]]
        updated_data.append(item)

    return updated_data

input_file = "td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.keypoints.json"
output_file = "person_keypoints_test-dev2017_ViTPose-small-256x192_results.json"

with open(input_file, "r") as infile:
    data = json.load(infile)

processed_data = process_json(data)

with open(output_file, "w") as outfile:
    json.dump(processed_data, outfile, indent=4)

print(f"File elaborato salvato in {output_file}")