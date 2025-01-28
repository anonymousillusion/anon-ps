from torch.utils.data import Dataset
import torchaudio
import json
import os
import torch
from tqdm.auto import tqdm

class AudioMetaDataset(Dataset):
    def __init__(self, audio_dir:str, metadata_file_path:str):
        """
        Args:
            audio_dir (string): Directory with all the audio files.
            metadata_file_path (string): Path to the JSON file with metadata.
        """
        self.audio_dir = audio_dir
        with open(metadata_file_path, 'r') as f:
            self.metadata = json.load(f)
        self.audio_files = list(self.metadata.keys())

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        # print(f"[LOG] audio_path: {audio_path}")
        audio_data, sample_rate = torchaudio.load(audio_path)
        
        metadata = self.metadata[audio_file]

        sample = {'audio': audio_data, 'sample_rate': sample_rate, 'metadata': metadata}

        return sample

# Example usage:
# dataset = AudioMetaDataset(audio_dir='path/to/audio', metadata_file='path/to/metadata.json')
# print(dataset[0])
def collate_fn(batch):
    max_audio_len = max([len(sample['audio'][0]) for sample in batch])
    for sample in batch:
        audio_len = len(sample['audio'][0])
        padding = torch.zeros(1, max_audio_len - audio_len)
        sample['audio'] = torch.cat((sample['audio'], padding), dim=1)
    return batch
if __name__ == "__main__":
    dataset = AudioMetaDataset(audio_dir='data/SLR103/Hindi_test/test/attack-a1', metadata_file_path='data/SLR103/Hindi_test/test/attack-a1-metadata.json')
    print(len(dataset))
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    
    for i in tqdm(dataloader,total=len(dataloader)):
        print(i)
        print(i[0]['audio'].shape)
        print(i[0]['metadata'])
        break
    
    print("Done!")
