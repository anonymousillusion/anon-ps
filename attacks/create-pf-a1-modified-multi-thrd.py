import pathlib
import soundfile as sf
import numpy as np
import torch
import random
from audio_gender_clf import get_gender
import torchaudio
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, read_audio, _, _) = utils

split = "test"

# save_path = f"/DATA1/ICASSP-2025/data/SLR104/Hindi-English/{split}/attack-a1-modified" # for SLR104
save_path = f"/DATA1/ICASSP-2025/data/SLR103/Hindi_{split}/{split}/attack-a1-modified" # for SLR103

Path(save_path).mkdir(parents=True, exist_ok=True) # create the save_path directory if it does not exist

# audio_paths = pathlib.Path(f"/DATA1/ICASSP-2025/data/SLR104/Hindi-English/{split}/segments") # for SLR104
audio_paths = pathlib.Path(f"/DATA1/ICASSP-2025/data/SLR103/Hindi_{split}/{split}/audio") # for SLR103
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)
print(f"[INFO] Number of audio files: {len(audio_files_list)}")


### for SLR104 ###
# audios_tts_bark_list_female = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-bark/").glob("female/*.wav"))
# audios_tts_bark_list_male = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-bark/").glob("male/*.wav"))
# # audios_tts_fbmms_list_female = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-fbmms/").glob("*.wav"))
# # audios_tts_fbmms_list_male = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-fbmms/").glob("*.wav"))
# audios_tts_f2hs_list_female = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-f2hs/").glob("female/*.wav"))
# audios_tts_f2hs_list_male = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-f2hs/").glob("male/*.wav"))
# audios_tts_f2mfa_list_female = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-f2mfa/").glob("female/*.wav"))
# audios_tts_f2mfa_list_male = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-f2mfa/").glob("male/*.wav"))
# audios_tts_indic_list_female = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-indic/").glob("female/*.wav"))
# audios_tts_indic_list_male = list(pathlib.Path(f"data/SLR104/Hindi-English/{split}/tts-indic/").glob("male/*.wav"))
### for SLR104 ###

### for SLR103 ###
audios_tts_bark_list_female = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-bark/").glob("female/*.wav"))
audios_tts_bark_list_male = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-bark/").glob("male/*.wav"))
audios_tts_fbmms_list_female = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fbmms/").glob("*.wav"))
audios_tts_fbmms_list_male = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fbmms/").glob("*.wav"))
audios_tts_f2hs_list_female = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fs2hs/").glob("female/*.wav"))
audios_tts_f2hs_list_male = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fs2hs/").glob("male/*.wav"))
audios_tts_f2mfa_list_female = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fs2mfa/").glob("female/*.wav"))
audios_tts_f2mfa_list_male = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-fs2mfa/").glob("male/*.wav"))
audios_tts_indic_list_female = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-indic/").glob("female/*.wav"))
audios_tts_indic_list_male = list(pathlib.Path(f"data/SLR103/Hindi_{split}/{split}/tts-indic/").glob("male/*.wav"))
### for SLR103 ###

# print(audio_files_list[:5])

tts_names = ["tts_bark", "tts_fbmms", "tts_f2hs", "tts_f2mfa", "tts_indic"] # for SLR103
# tts_names = ["tts_bark", "tts_f2hs", "tts_f2mfa", "tts_indic"] # for SLR104

tts_audios_dict = {
    "tts_bark": [audios_tts_bark_list_female, audios_tts_bark_list_male],
    "tts_fbmms": [audios_tts_fbmms_list_female, audios_tts_fbmms_list_male], # comment for SLR104
    "tts_f2hs": [audios_tts_f2hs_list_female, audios_tts_f2hs_list_male],
    "tts_f2mfa": [audios_tts_f2mfa_list_female, audios_tts_f2mfa_list_male],
    "tts_indic": [audios_tts_indic_list_female, audios_tts_indic_list_male]
}

# printing length of all audios in tts_audios_dict
for tts_name, audios in tts_audios_dict.items():
    print(f"[INFO] name : {tts_name}, len: {len(audios[0])}, {len(audios[1])}")

# num_of_tts_to_be_used = np.random.randint(1, 6)

metadata_dict1 = {}
metadata_dict2 = {}


def process_audio(audio):
    file_name = str(audio).split("/")[-1]
    # print(f"[LOG] Processing audio file: {file_name}")
    audio_metadata = {}
    # wav = read_audio(audio)
    
    y, sr = torchaudio.load(audio)
    y = torchaudio.functional.resample(y, sr, 16000)
    sr = 16000
    num_of_segments = np.random.randint(2, 10) # Note: Number of segments is randomly selected between 2 and 9 for test, else 5
    
    for i in range(num_of_segments):
        # speech_timestamps = get_speech_timestamps(wav, model)
        audio_metadata[f"segment_{i+1}"] = []
        window_length = np.random.randint(20, 640)
        range_value = int(window_length / 1000 * sr)
        range_upper_bound = min(range_value, len(y[0]))
        starting_timestamp = random.randint(0, len(y[0]) - sr) if (len(y[0]) - sr > 0) else 0
        
        # getting the gender of the original audio
        original_audio_gender = get_gender(audio)
        
        # getting the tts model to used in one segment
        num_of_tts_to_be_used = np.random.randint(1, 6)
        
        tts_seg_inc = range_upper_bound // num_of_tts_to_be_used
        
        for k in range(num_of_tts_to_be_used):
            tts_model_name = random.choice(tts_names)
            # print(f"[INFO] tts model name: {tts_model_name}")
            tts_audio_list = tts_audios_dict[tts_model_name][original_audio_gender]
            # print(f"[INFO] tts_audio_list[0]: {tts_audio_list[:2]}, {original_audio_gender}")
            # getting audio file thay has same filename as the original audio
            try:
                tts_audio = [audio for audio in tts_audio_list if file_name in str(audio).split('/')[-1]][0]
            except Exception as e:
                print(f"[ERROR] {e}, {file_name}")
                exit(0)
            y_fake, sr_fake = torchaudio.load(tts_audio)
            y_fake = torchaudio.functional.resample(y_fake, sr_fake, 16000)
            
            # Ensure y_fake has the same number of channels as y
            if y_fake.shape[0] != y.shape[0]:
                y_fake = y_fake.expand(y.shape[0], -1)

            # Ensure y_fake is long enough
            if y_fake.shape[1] < starting_timestamp + tts_seg_inc:
                # padding = starting_timestamp + tts_seg_inc - y_fake.shape[1]
                # y_fake = torch.nn.functional.pad(y_fake, (0, padding))
                y_fake = y_fake.repeat(1, 40)

            # Check if y_fake is still empty after the above operations
            if y_fake.shape[1] == 0:
                print(f"[ERROR] y_fake is empty after processing, skipping this segment.")
                print(f"[ERROR] File name: {file_name}")
            
            y_spoof = y.clone().detach()
            # print(f"[INFO] y_spoof: {y_spoof.shape}")
            try:
                y_spoof[:, starting_timestamp: (starting_timestamp + tts_seg_inc)] = y_fake[:, starting_timestamp: (starting_timestamp + tts_seg_inc)] * 0.25
                # y_spoof[0][starting_timestamp: starting_timestamp + range_upper_bound] = 0
            except Exception as e:
                print(f"[ERROR] y_spoof: {y_spoof.shape}, y_fake: {y_fake.shape}, starting_timestamp: {starting_timestamp}, tts_seg_inc: {tts_seg_inc}, added: {starting_timestamp + tts_seg_inc}")
                print(f"[ERROR] Error Log: {e}")
                print(f"[ERROR] File name: {file_name}")
                exit(1)
                
            y = y_spoof
            
            audio_metadata[f"segment_{i+1}"].append({    
                "tts_model_name": tts_model_name,
                "starting_timestamp": starting_timestamp,
                "ending_timestamp": starting_timestamp + range_upper_bound,
                "segement_length": range_upper_bound,
                "window_length_ms": window_length
            })
                
            starting_timestamp += tts_seg_inc
        
    
    torchaudio.save(f"{save_path}/{file_name}", y, 16000)
    
    return (file_name, audio_metadata)

# with ThreadPoolExecutor(max_workers=100) as executor:
#     results = list(tqdm(executor.map(process_audio, audio_files_list[:]), total=len(audio_files_list)))

# for file_name, metadata in results:
#     metadata_dict[file_name] = metadata

# # Save metadata_dict to a JSON file
# with open('data/SLR103/Hindi_{split}/{split}/attack-a1-modified-metadata.json', 'w') as f:
#     json.dump(metadata_dict, f)
    
with ThreadPoolExecutor(max_workers=100) as executor:
    audio_files_list_70_pec = audio_files_list[:int(len(audio_files_list)*0.7)]
    audio_files_list_30_pec = audio_files_list[int(len(audio_files_list)*0.7):]
    print(f"[INFO] audio_files_list_70_pec: {len(audio_files_list_70_pec)}, audio_files_list_30_pec: {len(audio_files_list_30_pec)}")
    results1 = list(tqdm(executor.map(process_audio, audio_files_list_70_pec), total=len(audio_files_list_70_pec)))
    results2 = list(tqdm(executor.map(process_audio, audio_files_list_30_pec), total=len(audio_files_list_30_pec)))

for file_name, metadata in results1:
   metadata_dict1[file_name] = metadata

for file_name, metadata in results2:
    metadata_dict2[file_name] = metadata

# Save metadata_dict to a JSON file
with open(f'{save_path}-70-per.json', 'w') as f:
   json.dump(metadata_dict1, f, indent=2)
    
with open(f'{save_path}-30-per.json', 'w') as f:
    json.dump(metadata_dict2, f, indent=2)
