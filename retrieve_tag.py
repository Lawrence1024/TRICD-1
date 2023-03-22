import requests
import json
url = 'https://raw.githubusercontent.com/ashkamath/TRICD/main/vqa/annotations/TRICD_VQA_val.json'
vqa_anns = json.loads(requests.get(url).text)
json_object = json.dumps(vqa_anns, indent=4)
with open("tags.json", "w") as outfile:
    outfile.write(json_object)
