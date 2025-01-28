import whisperx
import gc 
import pathlib
import librosa
import soundfile as sf
import numpy as np
import random

audio_paths = pathlib.Path("./data/svarah/audio")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)

# print(audio_files_list[2])
device = "cuda" 
# audio_file = audio_files_list[2]
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

for audio_file in audio_files_list[:5]:
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)
    print("[LOG] before alignment",result["segments"]) # before alignment
    asr_without_alignment = result["segments"]

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print("[LOG] after alignment",result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a



    # IN CASE of MULTI Speakers.
    # # 3. Assign speaker labels
    # diarize_model = whisperx.DiarizationPipeline(use_auth_token=YOUR_HF_TOKEN, device=device)

    # # add min/max number of speakers if known
    # diarize_segments = diarize_model(audio)
    # # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

    # result = whisperx.assign_word_speakers(diarize_segments, result)
    # print(diarize_segments)
    # print(result["segments"]) # segments are now assigned speaker IDs

    # Audio modification part
    y, sr = librosa.load(audio_file, sr=None) # None:- presever orignal Sampling rate
    print("sample points", len(y))
    print("audio duration", len(y)/sr)
    # print(sr)
    # print(y.shape)
    
    # selecting randomly replacing window.
    # max_possible_len = min(len(y),sr/1000*640)

    # random range selection
    range_value = int(np.random.randint(20,640)/1000 * sr)
    # range_value = 1 * sr # testing line to remove 1 sec
    range_upper_bound = min(range_value,len(y)) # in case range sleceted > audio length
    # print((len(y) - range_upper_bound))
    print("range to modify", range_upper_bound)
    
    starting_timestamp = np.random.randint(0,1+(len(y) - range_upper_bound)) # if no speech detected
    if len(asr_without_alignment): # if atleast one word is detected
        starting_timestamp = random.choice(random.choice(result["segments"])["words"])["start"] if len(result["segments"]) > 0 else  asr_without_alignment[0]["start"]
    
    starting_timestamp = int(starting_timestamp*sr)
    print("starting timestamp", starting_timestamp)

    y_spoof = y.copy()
    y_spoof[starting_timestamp: starting_timestamp + range_upper_bound] = 0
    
    sf.write("data/audios/original/"+str(audio_file).split("/")[-1], y, sr)
    sf.write("data/audios/spoofed/modified_" + str(audio_file).split("/")[-1], y_spoof, sr)
    print("-"*100)