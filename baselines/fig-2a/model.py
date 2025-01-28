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

# Max-Feature-Map which is based on Max-Out activation function. ref: https://arxiv.org/pdf/1904.05576
class MFM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MFM, self).__init__()
        self.out_channels = out_channels
    
    def forward(self, x):
        # Split the channels into two parts and take the max of each part
        out = torch.max(x[:, :self.out_channels, :, :], x[:, self.out_channels:, :, :])
        return out

class LCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fontend = DSPFrontEnd()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM(64, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm2 = MFM(64, 32)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MFM(96, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(48)
        
        self.conv4 = nn.Conv2d(48, 96, kernel_size=1, stride=1)
        self.mfm4 = MFM(96, 48)
        self.bn4 = nn.BatchNorm2d(48)
        
        self.conv5 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm5 = MFM(128, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.mfm6 = MFM(128, 64)
        self.bn6 = nn.BatchNorm2d(64)
        
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm7 = MFM(64, 32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn7 = nn.BatchNorm2d(32)
        
        self.conv8 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm8 = MFM(64,32)
        self.bn8 = nn.BatchNorm2d(32)
        
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm9 = MFM(64, 32)
        
        
    
    def forward(self, x):
        # x = self.fontend(x)  # [batch_size, 64, 863, 600]
        print("fontend", x.shape)
        x = self.conv1(x)  # [batch_size, 64, 863, 600]
        print("conv1", x.shape)
        x = self.mfm1(x)   # [batch_size, 32, 864, 600]
        print("mfm1", x.shape)
        # -----------------------------------------------
        
        x = self.pool1(x)  # [batch_size, 32, 431, 300]
        print("pool1", x.shape)
        # -----------------------------------------------
        
        x = self.conv2(x)  # [batch_size, 64, 431, 300]
        print("conv2", x.shape)
        x = self.mfm2(x)   # [batch_size, 32, 431, 300]
        print("mfm2", x.shape)
        x = self.bn2(x)    # [batch_size, 32, 431, 300]
        print("bn2", x.shape)
        x = self.conv3(x)  # [batch_size, 96, 431, 300]
        print("conv3", x.shape)
        x = self.mfm3(x)   # [batch_size, 48, 431, 300]
        print("mfm3", x.shape)
        # -----------------------------------------------
        
        x = self.pool2(x)  # [batch_size, 48, 215, 150]
        print("pool2", x.shape)
        x = self.bn3(x)    # [batch_size, 48, 215, 150]
        print("bn3", x.shape)
        # -----------------------------------------------
        
        x = self.conv4(x)  # [batch_size, 96, 215, 150]
        print("conv4", x.shape)
        x = self.mfm4(x)   # [batch_size, 48, 215, 150]
        print("mfm4", x.shape)
        x = self.bn4(x)    # [batch_size, 48, 215, 150]
        print("bn4", x.shape)
        x = self.conv5(x)  # [batch_size, 128, 215, 150]
        print("conv5", x.shape)
        x = self.mfm5(x)   # [batch_size, 64, 215, 150]
        print("mfm5", x.shape)
        # -----------------------------------------------
        
        x = self.pool3(x)  # [batch_size, 64, 107, 75]
        print("pool3", x.shape)
        # -----------------------------------------------
        
        # x = self.bn5(x)    # [batch_size, 64, 107, 75]
        
        x = self.conv6(x)  # [batch_size, 128, 107, 75]
        print("conv6", x.shape)
        x = self.mfm6(x)   # [batch_size, 64, 107, 75]
        print("mfm6", x.shape)
        x = self.bn6(x)    # [batch_size, 64, 107, 75]
        print("bn6", x.shape)
        x = self.conv7(x)  # [batch_size, 64, 107, 75]
        print("conv7", x.shape)
        x = self.mfm7(x)   # [batch_size, 32, 107, 75]
        print("mfm7", x.shape)
        x = self.bn7(x)    # [batch_size, 32, 107, 75]
        print("bn7", x.shape)
        x = self.conv8(x)  # [batch_size, 64, 107, 75]
        print("conv8", x.shape)
        x = self.mfm8(x)  # [batch_size, 32, 107, 75]
        print("mfm8", x.shape)
        x = self.bn8(x)  # [batch_size, 32, 107, 75]
        print("bn8", x.shape)
        x = self.conv9(x)  # [batch_size, 64, 107, 75]
        print("conv9", x.shape)
        x = self.mfm9(x) # [batch_size, 32, 107, 75]
        print("mfm9", x.shape)
        # -----------------------------------------------
        
        x = self.pool4(x) # [batch_size, 80, 53, 37]
        print("pool4", x.shape)
        
        return x
        

# SEBlock: Squeeze-and-Excitation block ref: https://arxiv.org/pdf/2107.14132
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SELCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MFM(64, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se1 = SEBlock(32)
        # -----------------------------------------------
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm2 = MFM(64, 32)
        self.bn2 = nn.BatchNorm2d(32)
        self.se2 = SEBlock(32)
        # -----------------------------------------------
        
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MFM(96, 48)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(48)
        self.se3 = SEBlock(48)
        # -----------------------------------------------
        
        self.conv4 = nn.Conv2d(48, 96, kernel_size=1, stride=1)
        self.mfm4 = MFM(96, 48)
        self.bn4 = nn.BatchNorm2d(48)
        self.se4 = SEBlock(48)
        # -----------------------------------------------
        
        self.conv5 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm5 = MFM(128, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.se5 = SEBlock(64)
        # -----------------------------------------------
        # self.bn5 = nn.BatchNorm2d(64)
        
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.mfm6 = MFM(128, 64)
        self.bn5 = nn.BatchNorm2d(64)
        self.se6 = SEBlock(64)
        # -----------------------------------------------
        
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm7 = MFM(64, 32)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn7 = nn.BatchNorm2d(32)
        self.se7 = SEBlock(32)
        # -----------------------------------------------
        
        self.conv8 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm8 = MFM(64,32)
        self.bn8 = nn.BatchNorm2d(32)
        self.se8 = SEBlock(32)
        # -----------------------------------------------
        
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm9 = MFM(64, 32)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        print("fontend", x.shape)
        x = self.conv1(x)
        print("conv1", x.shape)
        x = self.mfm1(x)
        print("mfm1", x.shape)
        x = self.pool1(x)
        print("pool1", x.shape)
        x = self.se1(x)
        print("se1", x.shape)
        # -----------------------------------------------
        
        x = self.conv2(x)
        print("conv2", x.shape)
        x = self.mfm2(x)
        print("mfm2", x.shape)
        x = self.bn2(x)
        print("bn2", x.shape)
        x = self.se2(x)
        print("se2", x.shape)
        # -----------------------------------------------
        
        x = self.conv3(x)
        print("conv3", x.shape)
        x = self.mfm3(x)
        print("mfm3", x.shape)
        x = self.pool2(x)
        print("pool2", x.shape)
        x = self.bn3(x)
        print("bn3", x.shape)
        x = self.se3(x)
        print("se3", x.shape)
        # -----------------------------------------------
        
        x = self.conv4(x)
        print("conv4", x.shape)
        x = self.mfm4(x)
        print("mfm4", x.shape)
        x = self.bn4(x)
        print("bn4", x.shape)
        x = self.se4(x)
        print("se4", x.shape)
        # -----------------------------------------------
        
        x = self.conv5(x)
        print("conv5", x.shape)
        x = self.mfm5(x)
        print("mfm5", x.shape)
        x = self.pool3(x)
        print("pool3", x.shape)
        x = self.se5(x)
        print("se5", x.shape)
        # -----------------------------------------------
        
        x = self.conv6(x)
        print("conv6", x.shape)
        x = self.mfm6(x)
        print("mfm6", x.shape)
        x = self.bn5(x)
        print("bn5", x.shape)
        x = self.se6(x)
        print("se6", x.shape)
        # -----------------------------------------------
        
        x = self.conv7(x)
        print("conv7", x.shape)
        x = self.mfm7(x)
        print("mfm7", x.shape)
        # x = self.pool4(x)
        x = self.bn7(x)
        print("bn7", x.shape)
        x = self.se7(x)
        print("se7", x.shape)
        # -----------------------------------------------
        
        x = self.conv8(x)
        print("conv8", x.shape)
        x = self.mfm8(x)
        print("mfm8", x.shape)
        x = self.bn8(x)
        print("bn8", x.shape)
        x = self.se8(x)
        print("se8", x.shape)
        # -----------------------------------------------
        
        x = self.conv9(x)
        print("conv9", x.shape)
        x = self.mfm9(x)
        print("mfm9", x.shape)
        x = self.pool4(x)
        print("pool4", x.shape)
        x = self.dropout(x)
        # -----------------------------------------------
        
        return x



# # Example usage
# model = LCNNModel()

# # Input example: [batch_size, channels, height, width]
# x = torch.randn(8, 1, 863, 600)
# print(x.shape)  # [8, 64, 863, 600]
# output = model(x)

# print(output.shape)  # [8, 2]


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
    waveform_length = 4*16000  # Example waveform length, e.g., 1 second of audio at 16kHz
    num_classes = config["model"]["num_classes"]

    # model = Model(input_channels=config["model"]["input_channels"], lcnn_channels=config["model"]["lcnn_channels"], num_classes=num_classes)
    model = SELCNN()
    # dummy_input = torch.randn(batch_size, waveform_length)  # Example input waveform
    dummy_input = torch.randn(batch_size, 64, 862, 600)  # Example input waveform
    output = model(dummy_input)
    print(output.shape)  # Expected output shape: (batch_size, num_classes)
    # print(output)
    # print(output.argmax(dim=-1))  # Predicted class labels
    print("Done!") 