import torch
import random
import torchaudio
from tqdm import tqdm
import json
import requests
from concurrent.futures import ThreadPoolExecutor

metadata_dict = {}


def fetch(url):
    response = requests.post(url, timeout=100000)
    return response.json()

fp = open("data/SLR104/Hindi-English/train/transcripts/text", "r")
transcribe_files_list = []
for line in fp.readlines():
    transcribe_files_list.append([line.split(" ")[0], (' '.join(line.split(" ")[1:]).removesuffix("\n"))])

def process_data(data):
    # print(f"[LOG] Processing {data}")
    file_name = data[0]
    text = data[1]
    url = f"http://10.6.0.22:80/lamam_3_1_instruct?prompt={text}"
    response = fetch(url)
    # print(f"[LOG] Response: {response}")
    # while response.get("status_code") != 200:
    #     print(f"[ERROR] Retrying {data}")
    #     response = await fetch(url)
    
    llm_response = response["output"]["content"]
    metadata = []
    
    metadata.append({
        "llm-response": llm_response
    })
    
    return (file_name, metadata)

def main():
    with ThreadPoolExecutor(max_workers=128) as executor:
        results = list(tqdm(executor.map(process_data, transcribe_files_list), total=len(transcribe_files_list[:])))
        
    # print(results)

    for file_name, metadata in results:
        metadata_dict[file_name] = metadata

    # Save metadata_dict to a JSON file
    with open('data/SLR104/Hindi-English/train/transcripts/llm-word-response.json', 'w') as f:
        json.dump(metadata_dict, f, ensure_ascii=False)
        
main()
print("Done!")