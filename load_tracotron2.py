import torch
import os

# Function to download and load the Tacotron2 model
def load_and_save_tacotron2_model(save_path,model, name='tacotron2'):
    print("Downloading and loading the model model...")

    # Load the Tacotron2 model from the Torch Hub
   
    
    # Move model to CUDA if available
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the model to evaluation mode
    model.eval()
    
    # Ensure the save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define the full path for saving the model
    model_save_path = os.path.join(save_path, f'{name}_model.pth')
    
    # Save the model checkpoint to the specified path
    torch.save(model.state_dict(), model_save_path)
    
    print(f"Model saved to {model_save_path}")
    return model

# Specify the path where you want to save the model
save_directory = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/pretrained/'

model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32',trust_repo=True, pretrianed = True)

# Call the function to load and save the model
model = load_and_save_tacotron2_model(save_directory, model, name='tacotron2')


model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp32',pretrianed = True,trust_repo=True)
model = load_and_save_tacotron2_model(save_directory, model , name='waveglow')