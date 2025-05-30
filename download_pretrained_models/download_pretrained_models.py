import os
import torch


# ===========================
# 💾 Save Model Utility
# ===========================
def load_and_save_model(model, save_path: str, name: str) -> torch.nn.Module:
    """
    Moves the model to the appropriate device, sets it to eval mode, and saves its state_dict.

    Args:
        model (torch.nn.Module): The model to process.
        save_path (str): Directory to save the model.
        name (str): Name to use for the saved .pth file.

    Returns:
        torch.nn.Module: The processed model.
    """
    print(f"📥 Processing model: {name}")

    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    # Ensure save path exists
    os.makedirs(save_path, exist_ok=True)

    # Save model checkpoint
    model_save_path = os.path.join(save_path, f'{name}_model.pth')
    torch.save(model.state_dict(), model_save_path)

    print(f"✅ Model '{name}' saved to: {model_save_path}")
    return model


# ===========================
# 🚀 Main Execution
# ===========================
if __name__ == "__main__":
    save_directory = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/pretrained/'

    # Load and save Tacotron2
    tacotron2 = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_tacotron2',
        model_math='fp32',
        pretrained=True,
        trust_repo=True
    )
    tacotron2 = load_and_save_model(tacotron2, save_directory, name='tacotron2')
    print("Tacotron2 model loaded and saved successfully.")
    # Load and save WaveGlow
    waveglow = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_waveglow',
        model_math='fp32',
        pretrained=True,
        trust_repo=True
    )
    waveglow = load_and_save_model(waveglow, save_directory, name='waveglow')
    print("WaveGlow model loaded and saved successfully.")