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

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, read_audio, _, _) = utils

save_path = "data/SLR104/Hindi-English/test/attack-a2"

audio_paths = pathlib.Path("data/SLR104/Hindi-English/test/segments")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)

audios_tts_bark_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-bark").glob("female/*.wav"))
audios_tts_bark_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-bark").glob("male/*.wav"))
# audios_tts_fbmms_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fbmms/").glob("*.wav"))
# audios_tts_fbmms_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fbmms/").glob("*.wav"))
audios_tts_f2hs_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2hs").glob("female/*.wav"))
audios_tts_f2hs_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2hs").glob("male/*.wav"))
audios_tts_f2mfa_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2mfa").glob("female/*.wav"))
audios_tts_f2mfa_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2mfa").glob("male/*.wav"))
audios_tts_indic_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-indic").glob("female/*.wav"))
audios_tts_indic_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-indic").glob("male/*.wav"))


# print(audio_files_list[:5])
# tts_names = ["tts_bark", "tts_fbmms", "tts_f2hs", "tts_f2mfa", "tts_indic"]
# tts_names = ["tts_bark", "tts_f2hs", "tts_f2mfa", "tts_indic"]
tts_names = ["tts_f2mfa", "tts_indic"]
tts_audios_dict = {
    "tts_bark": [audios_tts_bark_list_female, audios_tts_bark_list_male],
    # "tts_fbmms": [audios_tts_fbmms_list_female, audios_tts_fbmms_list_male],
    "tts_f2hs": [audios_tts_f2hs_list_female, audios_tts_f2hs_list_male],
    "tts_f2mfa": [audios_tts_f2mfa_list_female, audios_tts_f2mfa_list_male],
    "tts_indic": [audios_tts_indic_list_female, audios_tts_indic_list_male]
}

# num_of_tts_to_be_used = np.random.randint(1, 4)
# read_audio
metadata_dict1 = {}
metadata_dict2 = {}

with open('')


def process_audio(audio):
    file_name = str(audio).split("/")[-1]
    metadata = []
    # wav = read_audio(audio)
    try:
        y, sr = torchaudio.load(audio)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error: audio: {audio}")
    y = torchaudio.functional.resample(y, sr, 16000)
    sr = 16000
    # num_of_segments = np.random.randint(3, 7)
    
    for i in range(num_of_segments):
        # speech_timestamps = get_speech_timestamps(wav, model)
        window_length = np.random.randint(20, 640)
        range_value = int(window_length / 1000 * sr)
        range_upper_bound = min(range_value, len(y[0]))
        starting_timestamp = random.randint(0, len(y[0]) - min(sr, len(y[0])))
        
        tts_model_name = random.choice(tts_names)
        tts_audio_list = tts_audios_dict[tts_model_name][get_gender(audio)]
        tts_audio = [audio for audio in tts_audio_list if file_name in str(audio)][0]
        y_fake, sr_fake = torchaudio.load(tts_audio)
        y_fake = torchaudio.functional.resample(y_fake, sr_fake, 16000)
        
        if y_fake.shape[0] != y.shape[0]:
            y_fake = y_fake.expand(y.shape[0], -1)
        if y_fake.shape[1] < starting_timestamp + range_upper_bound:
            padding = starting_timestamp + range_upper_bound - y_fake.shape[1]
            y_fake = torch.nn.functional.pad(y_fake, (0, padding))
            # y_fake = torch.cat((y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0],y_fake[0])).unsqueeze(0)
        
        y_spoof = y.clone().detach()
        try:
            y_spoof[:, starting_timestamp: (starting_timestamp + range_upper_bound)] = y_fake[:, starting_timestamp: (starting_timestamp + range_upper_bound)] * 0.25
        except Exception as e:
            print(f"Error: file_name: {file_name}, starting_timestamp: {starting_timestamp}, range_upper_bound: {range_upper_bound}, y.shape: {y.shape}, y_fake.shape: {y_fake.shape} y_spoof.shape: {y_spoof.shape}")
            print("Error:", e)
            exit()
        y = y_spoof
        metadata.append({
            "tts_model_name": tts_model_name,
            "starting_timestamp": starting_timestamp,
            "ending_timestamp": starting_timestamp + range_upper_bound,
            "segement_length": range_upper_bound,
            "window_length_ms": window_length
        })
    
    torchaudio.save(f"{save_path}/{file_name}", y, 16000)
    
    return (file_name, metadata)

with ThreadPoolExecutor(max_workers=100) as executor:
    audio_files_list_70_pec = audio_files_list[:int(len(audio_files_list)*0.7)]
    audio_files_list_30_pec = audio_files_list[int(len(audio_files_list)*0.7):]
    results1 = list(tqdm(executor.map(process_audio, audio_files_list_70_pec), total=len(audio_files_list_70_pec)))
    results2 = list(tqdm(executor.map(process_audio, audio_files_list_30_pec), total=len(audio_files_list_30_pec)))

for file_name, metadata in results1:
    metadata_dict1[file_name] = metadata

for file_name, metadata in results2:
    metadata_dict2[file_name] = metadata

# Save metadata_dict to a JSON file
with open('data/SLR104/Hindi-English/test/transcripts/metadata-attack-a1-test-70-per.json', 'w') as f:
    json.dump(metadata_dict1, f, indent=2)
    
with open('data/SLR104/Hindi-English/test/transcripts/metadata-attack-a1-test-30-per.json', 'w') as f:
    json.dump(metadata_dict2, f, indent=2)