import torch, torchaudio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import json

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "quinnb/whisper-Large-v3-hindi"
model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=False, use_safetensors=True, cache_dir="./cache", force_download=False
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

# print(processor)

# audio_list = ["/DATA1/ICASSP-2025/CrisperWhisper/0116_008.wav","/DATA1/ICASSP-2025/CrisperWhisper/0116_003.wav"]
SPLIT = "eval"
BATCH_SIZE = 40 
dataset_split = "103_final_ps_database"
audio_path = Path(f"/DATA1/ICASSP-2025/Indic_PS_dataset/{dataset_split}/{SPLIT}")
audio_list = list(audio_path.glob("con_wav/*.wav"))

complete_response = []
for batch in tqdm(range(0, len(audio_list), BATCH_SIZE), desc=f"Processing {SPLIT}, of dataset{dataset_split}"):
    tensor_list = []
    for audio in audio_list[batch:batch+BATCH_SIZE]:
        y,sr = torchaudio.load(audio)
        # resample to 16k
        y = torchaudio.transforms.Resample(sr,16000)(y)
        tensor_list.append(y.squeeze().numpy())
    inputs = processor(tensor_list,sampling_rate=16_000)
    inputs["input_features"] = torch.from_numpy(np.array(inputs["input_features"])).to(device, dtype=torch.float16)
    inputs["language"] = "hi"
    outputs = model.generate(**inputs, )
    response = processor.batch_decode(outputs, skip_special_tokens=True)
    
    for idx,res in enumerate(response):
        complete_response.append(
            {
                "audio_id": str(audio_list[batch:batch+BATCH_SIZE][idx]),
                "transcript": res.encode("utf-8").decode("utf-8")
            }
        )
    
    Path(f"/DATA1/ICASSP-2025/Indic_PS_dataset/{dataset_split}/transcript").mkdir(parents=True, exist_ok=True)
            
    with open(f"/DATA1/ICASSP-2025/Indic_PS_dataset/{dataset_split}/transcript/{SPLIT}_{inputs['language']}.json", "w") as f:
        json.dump(complete_response, f, ensure_ascii=False, indent=4)


# tensor_list = []
# for audio in audio_list:
#     y,sr = torchaudio.load(audio)
#     # print(y.shape,sr)
#     tensor_list.append(y.squeeze().numpy())
# # print(tensor_list)
# inputs = processor(tensor_list)
# print(inputs)
# inputs["input_features"] = torch.from_numpy(np.array(inputs["input_features"])).to(device, dtype=torch.float16)
# # # for i in inputs:
# # #     print(f"input key: {i}, value: {inputs[i]}")
# outputs = model.generate(**inputs)
# print("-"*50)
# print(outputs)

# response = processor.batch_decode(outputs, skip_special_tokens=True)

# print(response)