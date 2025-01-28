import torchaudio
import glob
import pandas as pd
import os
from tqdm import tqdm
from math import ceil

audio_path = "data/SLR104/Hindi-English/test/audios/"
save_path = "data/SLR104/Hindi-English/test/segments/"

os.makedirs((save_path), exist_ok=True)

segement_text_file = "data/SLR104/Hindi-English/test/transcripts/segments"
df = pd.read_csv(segement_text_file, sep=" ", header=None, names=["audio_id", "file", "start", "end"], index_col=False)

for data in tqdm(df.iterrows(), total=len(df)):
    print("[LOG]", data[1]["audio_id"], data[1]["file"], data[1]["start"], data[1]["end"])
    
    audio, sr = torchaudio.load(audio_path+data[1]["file"]+".wav")
    new_audio = audio[:, int(data[1]["start"]*sr):int(ceil(data[1]["end"]*sr))]
    torchaudio.save(save_path+data[1]["audio_id"]+".wav", new_audio, sr)