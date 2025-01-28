import os
import sys
import torch, torchaudio

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from utils import adjust_pauses_for_hf_pipeline_output



device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "nyrahealth/CrisperWhisper"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, cache_dir="./cache"
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps='word',
    torch_dtype=torch_dtype,
    device=device,
)

# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

hf_pipeline_output = pipe("/DATA1/ICASSP-2025/CrisperWhisper/0116_008.wav")
crisper_whisper_result = adjust_pauses_for_hf_pipeline_output(hf_pipeline_output)
print(crisper_whisper_result)  