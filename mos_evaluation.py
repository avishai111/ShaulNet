import os
import argparse
import torch
import torchaudio
import distillmos

def main(audio_dir, device):
    # Initialize the Distill-MOS model
    model = distillmos.ConvTransformerSQAModel().to(device)
    model.eval()

    # Iterate over each WAV file in the directory
    for filename in os.listdir(audio_dir):
        if filename.lower().endswith(".wav"):
            filepath = os.path.join(audio_dir, filename)
            
            # Load the audio file
            waveform, sample_rate = torchaudio.load(filepath)
            
            # Convert to mono if necessary
            if waveform.shape[0] > 1:
                print(f"Warning!: {filename} has multiple channels, using only the first channel.")
                waveform = waveform[0, :]  # Select first channel
           
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            waveform = waveform.to(device)

            # Compute MOS
            with torch.no_grad():
                mos_score = model(waveform).item()
            
            print(f"{filename}: MOS = {mos_score:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MOS scores for WAV files in folder using Distill-MOS.")
    parser.add_argument("audio_dir", nargs="?", default="./outputs/", type=str, help="Directory containing WAV files (default: ./outputs/)")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run model on (default: auto-detect CUDA if available)")

    args = parser.parse_args()
    device = torch.device(args.device)
    main(args.audio_dir, device)
