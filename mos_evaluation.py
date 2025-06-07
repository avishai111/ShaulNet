import os
import torch
import torchaudio
import distillmos

# Initialize the Distill-MOS model
model = distillmos.ConvTransformerSQAModel()
model.eval()

# Directory containing your WAV files
audio_dir = "/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/outputs/"

# Iterate over each WAV file in the directory
for filename in os.listdir(audio_dir):
    if filename.lower().endswith(".wav"):
        filepath = os.path.join(audio_dir, filename)
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(filepath)
        
        # Convert to mono if necessary
        if waveform.shape[0] > 1:
            print(f"Warning: file has multiple channels, using only the first channel." )
        
        waveform = waveform[0, None, :]
        
             
        # Resample if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Compute MOS
        with torch.no_grad():
            mos_score = model(waveform).item()
        
        print(f"{filename}: MOS = {mos_score:.2f}")
        # generated.wav: MOS = 4.00
        # generated1.wav: MOS = 3.10
