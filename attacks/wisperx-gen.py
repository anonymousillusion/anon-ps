import whisperx
import gc 
import pathlib
import librosa
import soundfile as sf
import numpy as np
import random
import json
from tqdm.auto import tqdm
import torchaudio
import os

torchaudio.set_audio_backend("soundfile")

audio_paths = pathlib.Path("./data/SLR103/Hindi_train/train/audio")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)
data_save_path = "./data/SLR103/Hindi_train/train/audio_transcripts.json"
# print(audio_files_list[2])
device = "cuda" 
# audio_file = audio_files_list[2]
batch_size = 16 # reduce if low on GPU mem
compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v3", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)
# model_a, metadata = whisperx.load_align_model(language_code="hi", device=device)
data_output = {} if not pathlib.Path(data_save_path).exists() else json.load(open(data_save_path))["word_transcribe"]


for audio_file in tqdm(audio_files_list, total=len(audio_files_list)):
    if data_output.get(str(audio_file).split("/")[-1].replace(".wav","")):
        print("[INFO] Skipping audio file:", audio_file)
        continue
    
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="hi")
    # print("[LOG] before alignment",result["segments"]) # before alignment
    asr_without_alignment = result["segments"]

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    data_output[str(audio_file).split("/")[-1].replace(".wav","")] = result["segments"]
    
    # saving intermediate results
    if len(data_output) % 100 == 0:
        json.dump({"word_transcribe": data_output}, open(data_save_path, "w"), ensure_ascii=False)
    # print("[INFO] audio file:", audio_file)
    # print("[LOG] after alignment",result["segments"]) # after alignment
    # print("-"*50)


# print(data_output)
# saving complete results
json.dump({"word_transcribe": data_output}, open(data_save_path, "w"), ensure_ascii=False)