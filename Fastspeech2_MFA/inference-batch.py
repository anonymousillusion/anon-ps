# import os

# # python3 -m TTS.bin.synthesize --text "hello world my name is jupyter" \\n    --model_path ./hi/fastpitch/best_model.pth \\n    --config_path ./hi/fastpitch/config.json \\n    --vocoder_path ./hi/hifigan/best_model.pth \\n    --vocoder_config_path ./hi/hifigan/config.json \\n    --out_path test_audio.wav

# import soundfile as sf
import pathlib, glob
from tqdm.auto import tqdm
# import random
# import os
# from concurrent.futures import ThreadPoolExecutor


saved_audio_path = "../data/SLR103/Hindi_train/train/tts-fs2mfa/female"
gen_audio_lsit = [_.split("/")[-1].replace(".wav","") for _ in glob.glob(f"{saved_audio_path}/*.wav")]
txt_file_path = "../data/SLR103/Hindi_train/train/transcription.txt"
batch_size = 30
j = 0

# gen_command = """
# CUDA_VISIBLE_DEVICES=0 \
# python3 inference.py \
# --sample_text "{}" \
# --language Hindi \
# --gender male \
# --output_file {}"""


            
# def execute_command(t_txt, t_file_id, saved_audio_path, gen_command):
#     cmd = gen_command.format(t_txt, f"{saved_audio_path}/{t_file_id}.wav")
#     # print(f'cmd generated: {cmd}')
#     os.system(cmd)

# # loading file
# with open(txt_file_path,"r") as fp:
#     lines = fp.readlines()
#     # random.shuffle(lines)
#     with tqdm(total=len(lines)) as pbar:
#         while j < len(lines):
#             t_txt_list = []
#             t_file_id_list = []
#             for line in lines[j:j+batch_size]:
#                 splited = line.split()
#                 file_id, transcribe = splited[0], " ".join(splited[1:])
#                 # print(file_id,transcribe)
#                 t_txt_list.append(transcribe)
#                 t_file_id_list.append(file_id)

#             if set(t_file_id_list).issubset(gen_audio_lsit):
#                 pbar.update(batch_size)
#                 j += batch_size
#                 print(f"audio already generated {t_file_id_list}")
#                 continue
            
#             # generating audio
#             with ThreadPoolExecutor() as executor:
#                 futures = [
#                     executor.submit(execute_command, t_txt_list[idx], t_file_id, saved_audio_path, gen_command)
#                     for idx, t_file_id in enumerate(t_file_id_list)
#                 ]
            
#             # saving file
#             # for idx,t_file_id in enumerate(t_file_id_list):
#             #     sf.write(f"data/SLR103/Hindi_test/test/tts-bark/male/{t_file_id}.wav", speech_output[idx], sampling_rate)
            
#             print(f"[INFO] percentage done: {j/len(lines) * 100}, j/total: {j}/{len(lines)}")
#             pbar.update(batch_size)
#             j += batch_size
#             if pbar.n > len(lines):
#                 break
# print("ALL DONE !!!")



import sys
import os
#replace the path with your hifigan path to import Generator from models.py 
sys.path.append("hifigan")
import argparse
import torch
from espnet2.bin.tts_inference import Text2Speech
from models import Generator
from scipy.io.wavfile import write
from meldataset import MAX_WAV_VALUE
from env import AttrDict
import json
import yaml
from text_preprocess_for_inference import TTSDurAlignPreprocessor

SAMPLING_RATE = 22050

def load_hifigan_vocoder(language, gender, device):
    # Load HiFi-GAN vocoder configuration file and generator model for the specified language and gender
    vocoder_config = f"vocoder/{gender}/aryan/hifigan/config.json"
    vocoder_generator = f"vocoder/{gender}/aryan/hifigan/generator"
    # Read the contents of the vocoder configuration file
    with open(vocoder_config, 'r') as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    # Move the generator model to the specified device (CPU or GPU)
    device = torch.device(device)
    generator = Generator(h).to(device)
    state_dict_g = torch.load(vocoder_generator, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    # Return the loaded and prepared HiFi-GAN generator model
    return generator


def load_fastspeech2_model(language, gender, device):
    
    #updating the config.yaml fiel based on language and gender
    with open(f"{language}/{gender}/model/config.yaml", "r") as file:      
     config = yaml.safe_load(file)
    
    current_working_directory = os.getcwd()
    feat="model/feats_stats.npz"
    pitch="model/pitch_stats.npz"
    energy="model/energy_stats.npz"
    
    feat_path=os.path.join(current_working_directory,language,gender,feat)
    pitch_path=os.path.join(current_working_directory,language,gender,pitch)
    energy_path=os.path.join(current_working_directory,language,gender,energy)

    
    config["normalize_conf"]["stats_file"]  = feat_path
    config["pitch_normalize_conf"]["stats_file"]  = pitch_path
    config["energy_normalize_conf"]["stats_file"]  = energy_path
        
    with open(f"{language}/{gender}/model/config.yaml", "w") as file:
        yaml.dump(config, file)
    
    tts_model = f"{language}/{gender}/model/model.pth"
    tts_config = f"{language}/{gender}/model/config.yaml"
    
    
    return Text2Speech(train_config=tts_config, model_file=tts_model, device=device)

def text_synthesis(language, gender, sample_text, vocoder, MAX_WAV_VALUE, device):
    # Perform Text-to-Speech synthesis
    with torch.no_grad():
        # Load the FastSpeech2 model for the specified language and gender
        
        model = load_fastspeech2_model(language, gender, device)
        audios = []
        for text in sample_text:
            # Generate mel-spectrograms from the input text using the FastSpeech2 model
            out = model(text, decode_conf={"alpha": 1})
            print("TTS Done")  
            x = out["feat_gen_denorm"].T.unsqueeze(0) * 2.3262
            # x = out["feat_gen_denorm"].T * 2.3262
            x = x.to(device)
            
            # Use the HiFi-GAN vocoder to convert mel-spectrograms to raw audio waveforms
            y_g_hat = vocoder(x)
            audio = y_g_hat.squeeze()
            audio = y_g_hat
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            audios.append(audio)
            # Return the synthesized audio
        return audios


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Text-to-Speech Inference")
    language = "Hindi"
    gender = "female"
    
    # parser.add_argument("--language", type=str, required=True, help="Language (e.g., hindi)")
    # parser.add_argument("--gender", type=str, required=True, help="Gender (e.g., female)")
    # parser.add_argument("--sample_text", type=str, required=True, help="Text to be synthesized")
    # parser.add_argument("--output_file", type=str, default="", help="Output WAV file path")

    # args = parser.parse_args()
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the HiFi-GAN vocoder with dynamic language and gender
    vocoder = load_hifigan_vocoder(language, gender, device)
    preprocessor = TTSDurAlignPreprocessor()

    with open(txt_file_path,"r") as fp:
        lines = fp.readlines()
        # Preprocess the sample text
        
        with tqdm(total=len(lines)) as pbar:
            while j < len(lines):
                t_txt_list = []
                t_file_id_list = []
                for line in lines[j:j+batch_size]:
                    splited = line.split()
                    file_id, transcribe = splited[0], " ".join(splited[1:])
                    # print(file_id,transcribe)
                    t_txt_list.append(transcribe)
                    t_file_id_list.append(file_id)

                if set(t_file_id_list).issubset(gen_audio_lsit):
                    pbar.update(batch_size)
                    j += batch_size
                    print(f"audio already generated {t_file_id_list}")
                    continue
                
                preprocessed_text_list = []
                for sample_text in t_txt_list:
                    preprocessed_text, phrases = preprocessor.preprocess(sample_text, language, gender)
                    preprocessed_text = " ".join(preprocessed_text)
                    preprocessed_text_list.append(preprocessed_text)
            
                audios = text_synthesis(language, gender, preprocessed_text_list, vocoder, MAX_WAV_VALUE, device)
            
                for idx, audio in enumerate(audios):
                    write(f"{saved_audio_path}/{t_file_id_list[idx]}.wav", SAMPLING_RATE, audio)
                pbar.update(batch_size)
                j += batch_size