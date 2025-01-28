import pathlib
import soundfile as sf
import numpy as np
import pandas as pd
import torch
import random
# from audio_gender_clf import get_gender
import torchaudio
from tqdm import tqdm
import json
from concurrent.futures import ThreadPoolExecutor






##################################################################################################################


import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)


class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender



# load model from hub
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = AgeGenderModel.from_pretrained(model_name).to(device)

# dummy signal

# sampling_rate=16000

def process_func(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict age and gender or extract embeddings from raw audio signal."""

    # run through processor to normalize signal
    # always returns a batch, so we just get the first entry
    # then we put it on the device
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = y.reshape(1, -1)
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)
        if embeddings:
            y = y[0]
        else:
            y = torch.hstack([y[1], y[2]])

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

@torch.inference_mode
def get_gender(audio_path: str) -> int:
    """
    return 0 if female, 1 if male, 2 if child
    """
    audio, sampling_rate = torchaudio.load(audio_path)
    resample_rate = 16000
    resampler = T.Resample(sampling_rate, resample_rate, dtype=audio.dtype)
    signal = resampler(audio[:,:resample_rate*15])
    return process_func(signal, resample_rate)[0][1:3].argmax()


##################################################################################################################











# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, read_audio, _, _) = utils

save_path = "data/SLR104/Hindi-English/test/attack-a1-modified"

pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)

audio_paths = pathlib.Path("./data/SLR104/Hindi-English/test/audios/")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)

audios_tts_bark_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-bark/").glob("female/*.wav"))
audios_tts_bark_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-bark/").glob("male/*.wav"))
# audios_tts_fbmms_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-fbmms/").glob("*.wav"))
# audios_tts_fbmms_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-fbmms/").glob("*.wav"))
audios_tts_f2hs_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2hs").glob("female/*.wav"))
audios_tts_f2hs_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2hs").glob("male/*.wav"))
audios_tts_f2mfa_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2mfa").glob("female/*.wav"))
audios_tts_f2mfa_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-f2mfa").glob("male/*.wav"))
audios_tts_indic_list_female = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-indic/").glob("female/*.wav"))
audios_tts_indic_list_male = list(pathlib.Path("data/SLR104/Hindi-English/test/tts-indic/").glob("male/*.wav"))


print(audios_tts_f2mfa_list_male[:5])

tts_names = ["tts_bark", "tts_f2hs", "tts_f2mfa", "tts_indic"]
tts_audios_dict = {
    "tts_bark": [audios_tts_bark_list_female, audios_tts_bark_list_male],
    # "tts_fbmms": [audios_tts_fbmms_list_female, audios_tts_fbmms_list_male],
    "tts_f2hs": [audios_tts_f2hs_list_female, audios_tts_f2hs_list_male],
    "tts_f2mfa": [audios_tts_f2mfa_list_female, audios_tts_f2mfa_list_male],
    "tts_indic": [audios_tts_indic_list_female, audios_tts_indic_list_male]
}

num_of_tts_to_be_used = np.random.randint(1, 5)

metadata_dict = {}

audio_to_parts_txt = "data/SLR104/Hindi-English/test/transcripts/segments"
audio_to_parts_df = pd.read_csv(audio_to_parts_txt, sep=" ", header=None, names=["audio_id", "file_name", "start", "end"])
print(audio_to_parts_df.head())


def process_audio(audio):
    file_name = str(audio).split("/")[-1]
    
    
    audios_name_from_df = audio_to_parts_df.loc[audio_to_parts_df["file_name"] == file_name.replace(".wav",""),:]
    # print(file_name)
    # print(audios_name_from_df)
    y_original, sr = torchaudio.load(audio)
    y_original = torchaudio.functional.resample(y_original, sr, 16000)
    sr = 16000
    print(f"[INFO] y_original: {y_original.shape}")
    
    # getting the gender of the original audio
    original_audio_gender = get_gender(audio)
            
    audio_split_metadata = {}
    for index, row in tqdm(audios_name_from_df.iterrows(), total=audios_name_from_df.shape[0], desc=f"Processing {file_name}"):
        # print(row["start"], row["end"])
        print("-"*50)
        print(f"[INFO] audio_id: {row['audio_id']}")
        print("-"*50)
        start = row["start"]
        end = row["end"]
        audio_file_id = row['audio_id']
                
        audio_metadata = {}
        
        # wav = read_audio(audio)
        
        
        y = y_original[:,int(start*sr):int(end*sr)]
        print(f"[INFO] y: {y.shape}")
        num_of_segments = np.random.randint(2, 6)
        
        for i in range(num_of_segments):
            # speech_timestamps = get_speech_timestamps(wav, model)
            audio_metadata[f"segment_{i+1}"] = []
            window_length = np.random.randint(20, 640)
            range_value = int(window_length / 1000 * sr)
            range_upper_bound = min(range_value, len(y[0]))
            starting_timestamp = random.randint(0, len(y[0]) - sr)
            
            
            
            # getting the tts model to used in one segment
            num_of_tts_to_be_used = np.random.randint(1, 6)
            
            tts_seg_inc = range_upper_bound // num_of_tts_to_be_used
            
            for k in range(num_of_tts_to_be_used):
                tts_model_name = random.choice(tts_names)
                print(f"[INFO] tts model name: {tts_model_name}, gender: {original_audio_gender}")
                tts_audio_list = tts_audios_dict[tts_model_name][original_audio_gender]
            
                # getting audio file thay has same filename as the original audio
                tts_audio = [audio for audio in tts_audio_list if audio_file_id in str(audio)][0]
                y_fake, sr_fake = torchaudio.load(tts_audio)
                y_fake = torchaudio.functional.resample(y_fake, sr_fake, 16000)
                # print(f"[INFO] y_fake: {y_fake}, y : {y}")
                # print(f"[LOG] starting_timestamp: starting_timestamp + range_upper_bound: {starting_timestamp} : {starting_timestamp + range_upper_bound}")
            
                
                # Ensure y_fake has the same number of channels as y
                if y_fake.shape[0] != y.shape[0]:
                    y_fake = y_fake.expand(y.shape[0], -1)

                # Ensure y_fake is long enough
                if y_fake.shape[1] < starting_timestamp + tts_seg_inc:
                    # padding = starting_timestamp + tts_seg_inc - y_fake.shape[1]
                    # y_fake = torch.nn.functional.pad(y_fake, (0, padding))
                    y_fake = torch.cat((y_fake[0], y_fake[0],y_fake[0],y_fake[0],y_fake[0])).unsqueeze(0)

            
                y_spoof = y.clone().detach()
                # print(f"[INFO] y_spoof: {y_spoof.shape}")
                y_spoof[:,starting_timestamp: (starting_timestamp + tts_seg_inc)] = y_fake[:,starting_timestamp: (starting_timestamp + tts_seg_inc)] *0.25
                # y_spoof[0][starting_timestamp: starting_timestamp + range_upper_bound] = 0
                y = y_spoof
                
                audio_metadata[f"segment_{i+1}"].append({    
                    "tts_model_name": tts_model_name,
                    "starting_timestamp": starting_timestamp,
                    "ending_timestamp": starting_timestamp + range_upper_bound,
                    "segement_length": range_upper_bound,
                    "window_length_ms": window_length
                })
                    
                starting_timestamp += tts_seg_inc
            
        
            torchaudio.save(f"{save_path}/{row['audio_id']}.wav", y, 16000)
            audio_split_metadata[row["audio_id"]] = audio_metadata
        
    return (file_name, audio_split_metadata)

with ThreadPoolExecutor(max_workers=1) as executor:
    results = list(tqdm(executor.map(process_audio, audio_files_list[:]), total=len(audio_files_list)))

try:
    for file_name, metadata in results:
        metadata_dict[file_name.removesuffix(".wav")] = metadata
except Exception as e:
    print(f"Error: {e} & results: {results}")
# Save metadata_dict to a JSON file
with open(f'{save_path}.json', 'w') as f:
    json.dump(metadata_dict, f, indent=4)