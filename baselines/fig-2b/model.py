import torch
import torch.nn as nn
import torchaudio.transforms as transforms

class DSPFrontEnd(nn.Module):
    def __init__(self, n_fft=400, hop_length=160):
        super().__init__()
        # Using a basic spectrogram as an example of DSP-based front-end
        self.spectrogram = transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    
    def forward(self, x):
        # Input shape: (Batch, Time) -> (Batch, Freq, Time)
        # Convert waveform x to spectrogram
        x = self.spectrogram(x)
        return x

class LCNN(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.lcnn = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding), # First convolutional layer
            nn.ReLU(), 
            nn.Conv2d(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding), # Second convolutional layer
            nn.ReLU() # ReLU activation function
        )
    
    def forward(self, x):
        # Input shape: (Batch, Channels, Freq, Time) -> Output shape: (Batch, Channels, Freq, Time)
        x = self.lcnn(x)
        return x

class Model(nn.Module):
    def __init__(self, input_channels=1, lcnn_channels=16, linear1_out_features=2, linear2_out_features=10):
        super(Model, self).__init__()
        # DSP-based front-end: Feature extraction (e.g., spectrogram)
        self.dsp_frontend = DSPFrontEnd()
        # Layered Convolutional Neural Network
        self.lcnn = LCNN(input_channels, lcnn_channels)
        # Adaptive average pooling layer
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        # Flatten layer
        self.flatten = nn.Flatten()
   
        # Fully connected layer to map features to class scores
        self.fc1 = nn.Linear(lcnn_channels, linear1_out_features)
        self.fc2 = nn.Linear(lcnn_channels, linear2_out_features)
        
    def forward(self, x):
        # Input shape: (Batch, Time)
        x = self.dsp_frontend(x)
        # After DSP front-end: (Batch, Freq, Time)
        x = x.unsqueeze(1)  # Add channel dimension -> (Batch, 1, Freq, Time)
        x = self.lcnn(x)
        # After LCNN: (Batch, Channels, Freq, Time)
        x = self.pooling(x)
        # After pooling: (Batch, Channels, 1, 1)
        # x = x.view(x.size(0), -1)  # Flatten -> (Batch, Channels)
        x = self.flatten(x) # Flatten -> (Batch, Channels)
        x1 = self.fc1(x)
        x2 = self.fc2(x) 
        # Output shape: x1: (Batch, num_classes), x2: (Batch, features)
        return x1, x2


# Example usage
if __name__ == "__main__":
    # load ymal file
    import yaml
    with open("config.yaml", 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    batch_size = config["params"]["batch_size"]
    waveform_length = 16000  # Example waveform length, e.g., 1 second of audio at 16kHz
    num_classes = config["model"]["num_classes"]
    features_dim = config["model"]["features_dim"]

    model = Model(input_channels=config["model"]["input_channels"], lcnn_channels=config["model"]["lcnn_channels"], linear1_out_features=num_classes, linear2_out_features=features_dim)
    dummy_input = torch.randn(batch_size, waveform_length)  # Example input waveform
    output1, output2 = model(dummy_input)
    print("-"*50,"Model Output 1","-"*50)
    print(output1.shape)  # Expected output1 shape: (batch_size, num_classes)
    # print(output1)
    print(output1.argmax(dim=-1))  # Predicted class labels
    print("-"*50,"Model Output 2","-"*50)
    print(output2.shape)  # Expected output1 shape: (batch_size, features)
    print(output2.min())
    print(output2.max())
    
    print("Done!")