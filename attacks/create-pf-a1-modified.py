import pathlib
import soundfile as sf
import numpy as np
import torch
import random
from audio_gender_clf import get_gender
import torchaudio
from tqdm import tqdm
import json

# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, read_audio, _, _) = utils


save_path = "data/SLR103/Hindi_test/test/attack-a1-modified"

audio_paths = pathlib.Path("./data/SLR103/Hindi_test/test/audio")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)

audios_tts_bark_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-bark/").glob("female/*.wav"))
audios_tts_bark_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-bark/").glob("male/*.wav"))
audios_tts_fbmms_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fbmms/").glob("*.wav"))
audios_tts_fbmms_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fbmms/").glob("*.wav"))
audios_tts_f2hs_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fs2hs/").glob("female/*.wav"))
audios_tts_f2hs_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fs2hs/").glob("male/*.wav"))
audios_tts_f2mfa_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fs2mfa/").glob("female/*.wav"))
audios_tts_f2mfa_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-fs2mfa/").glob("male/*.wav"))
audios_tts_indic_list_female = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-indic/").glob("female/*.wav"))
audios_tts_indic_list_male = list(pathlib.Path("data/SLR103/Hindi_test/test/tts-indic/").glob("male/*.wav"))

print(f"[INFO] number of audio files loaded : {audios_tts_f2hs_list_female[:5]}")

tts_names = ["tts_bark", "tts_fbmms", "tts_f2hs", "tts_f2mfa", "tts_indic"]
tts_audios_dict = {
    "tts_bark": [audios_tts_bark_list_female, audios_tts_bark_list_male],
    "tts_fbmms": [audios_tts_fbmms_list_female, audios_tts_fbmms_list_male],
    "tts_f2hs": [audios_tts_f2hs_list_female, audios_tts_f2hs_list_male],
    "tts_f2mfa": [audios_tts_f2mfa_list_female, audios_tts_f2mfa_list_male],
    "tts_indic": [audios_tts_indic_list_female, audios_tts_indic_list_male]
}

# num_of_tts_to_be_used = np.random.randint(1, 6)

metadata_dict = {}

for audio in tqdm(audio_files_list[:]):
    file_name = str(audio).split("/")[-1]
    metadata_dict[file_name] = []
    # print(f"[INFO] processing file: {file_name}")
    # reading audio
    # wav = read_audio(audio)
    
    y, sr = torchaudio.load(audio) # None:- presever orignal Sampling rate
    y = torchaudio.functional.resample(y, sr, 16000)
    sr = 16000
    # selecting number of segeements to be replaced
    num_of_segments = np.random.randint(2, 6)
    
        
    for i in range(num_of_segments):
        # speech_timestamps = get_speech_timestamps(wav, model)
        # print(f"[INFO] speech_timestamps: {speech_timestamps}")
        # starting_timestamp = random.choice(speech_timestamps)['start']
        
        
        # random range selection
        range_value = int(np.random.randint(20,640)/1000 * sr)
        # range_value =  1 * sr # testing line to remove 1 sec
        range_upper_bound = min(range_value,len(y[0])) # in case range sleceted > audio length
        
        starting_timestamp = random.randint(0, len(y[0]) - sr)
        
        # getting the gender of the original audio
        original_audio_gender = get_gender(audio)
        
        # getting the tts model to used in one segment
        num_of_tts_to_be_used = np.random.randint(1, 6)
        
        tts_seg_inc = range_upper_bound // num_of_tts_to_be_used
        
        for i in range(num_of_tts_to_be_used):
            tts_model_name = random.choice(tts_names)
        # print(f"[INFO] tts model name: {tts_model_name}")
            tts_audio_list = tts_audios_dict[tts_model_name][original_audio_gender]
        
            # getting audio file thay has same filename as the original audio
            tts_audio = [audio for audio in tts_audio_list if file_name in str(audio)][0]
            y_fake, sr_fake = torchaudio.load(tts_audio)
            y_fake = torchaudio.functional.resample(y_fake, sr_fake, 16000)
            # print(f"[INFO] y_fake: {y_fake}, y : {y}")
            # print(f"[LOG] starting_timestamp: starting_timestamp + range_upper_bound: {starting_timestamp} : {starting_timestamp + range_upper_bound}")
        
            
            # Ensure y_fake has the same number of channels as y
            if y_fake.shape[0] != y.shape[0]:
                y_fake = y_fake.expand(y.shape[0], -1)

            # Ensure y_fake is long enough
            if y_fake.shape[1] < starting_timestamp + tts_seg_inc:
                padding = starting_timestamp + tts_seg_inc - y_fake.shape[1]
                y_fake = torch.nn.functional.pad(y_fake, (0, padding))

        
            y_spoof = y.clone().detach()
            # print(f"[INFO] y_spoof: {y_spoof.shape}")
            y_spoof[:,starting_timestamp: (starting_timestamp + tts_seg_inc)] = y_fake[:,starting_timestamp: (starting_timestamp + tts_seg_inc)] *0.25
            # y_spoof[0][starting_timestamp: starting_timestamp + range_upper_bound] = 0
            y = y_spoof
            metadata_dict[file_name].append( {
                "tts_model_name": tts_model_name,
                "starting_timestamp": starting_timestamp,
                "ending_timestamp": starting_timestamp + range_upper_bound
            })
            
            starting_timestamp += tts_seg_inc
        
    torchaudio.save(f"{save_path}/{str(audio).split('/')[-1]}", y_spoof, 16000)


print(f"[INFO] metadata_dict: {metadata_dict}")
json.dump({"spoofed_segement":metadata_dict}, open("data/SLR103/Hindi_test/test/attack-a1-modified-metadata.json", "w"))