"""
Attack A1, replacing random speech part in range from 20ms - 640 ms
"""
import pathlib
import librosa
import soundfile as sf
import numpy as np

audio_paths = pathlib.Path("./data/svarah/audio")
audio_files = audio_paths.glob("*.wav")
audio_files_list = list(audio_files)

print(f"[INFO] number of audio files loaded : {len(audio_files_list)}")

print(audio_files_list[:5])
for audio in audio_files_list[:5]:
    # reading audio
    y, sr = librosa.load(audio, sr=None) # None:- presever orignal Sampling rate
    print("sample points", len(y))
    print("audio duration", len(y)/sr)
    # print(sr)
    # print(y.shape)
    
    # selecting randomly replacing window.
    # max_possible_len = min(len(y),sr/1000*640)

    # random range selection
    range_value = int(np.random.randint(20,640)/1000 * sr)
    # range_value = 1 * sr # testing line to remove 1 sec
    range_upper_bound = min(range_value,len(y))
    # print((len(y) - range_upper_bound))
    print("range to modify", range_upper_bound)
    starting_timestamp = np.random.randint(0,1+(len(y) - range_upper_bound))

    y_spoof = y.copy()
    y_spoof[starting_timestamp: starting_timestamp + range_upper_bound] = 0
    
    sf.write(str(audio).split("/")[-1], y, sr)
    sf.write("modified_" + str(audio).split("/")[-1], y_spoof, sr)