import torch
import os
import sys
import argparse
import datetime
from typing import Union, Dict, List, Optional, Tuple, Any, Callable, Type
import warnings
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import hydra

# Preprocess imports
from Data_Creation_scripts.HebrewToEnglish import HebrewToEnglish

# === BigVGAN ===
from BigVGAN.bigvgan import BigVGAN

# === Matcha ===
from Matcha_TTS.matcha.models.matcha_tts import MatchaTTS
from Matcha_TTS.matcha.hifigan.models import Generator as MatchaVocoder
from Matcha_TTS.matcha.hifigan.config import v1 as matcha_hifigan_config
from Matcha_TTS.matcha.hifigan.denoiser import Denoiser as MatchaDenoiser
from Matcha_TTS.matcha.text import text_to_sequence as matcha_text_to_sequence
from Matcha_TTS.matcha.text import sequence_to_text as matcha_sequence_to_text
from Matcha_TTS.matcha.utils.utils import intersperse

# === Tacotron2 ===
from speechbrain.inference.TTS import Tacotron2 as Tacotron2
from speechbrain.inference.vocoders import HIFIGAN as SBHifiGAN
from tacotron2 import text_to_sequence as tacotron_text_to_sequence
from tacotron2 import load_model
from tacotron2 import create_hparams

# === Ringformer ===
from RingFormer import get_hparams_from_file, load_checkpoint, SynthesizerTrn

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# === Plotting ===
def plot_mel(mel: Union[torch.Tensor, 'np.ndarray'], title: str = "Mel-Spectrogram") -> None:
    """
    Plots a mel-spectrogram.

    Args:
        mel (np.ndarray or torch.Tensor): Mel spectrogram to plot, shape [n_mels, time].
        title (str): Title of the plot.
    """
    if torch.is_tensor(mel):
        mel = mel.detach().cpu().numpy().squeeze()

    plt.figure(figsize=(10, 4))
    plt.imshow(mel, aspect='auto', origin='lower', interpolation='none')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Mel Bin')
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.show()

# === MatchaTTS model inference ===

@torch.inference_mode()
def process_text(text: str, device: torch.device) -> Dict[str, Union[torch.Tensor, str]]:
    """
    Converts raw text into a tokenized tensor input for Matcha-TTS and extracts metadata.

    Args:
        text (str): The input text to be processed.
        device (torch.device): The target device for the resulting tensors (e.g., CPU or CUDA).

    Returns:
        Dict[str, Union[torch.Tensor, str]]: A dictionary containing:
            - 'x_orig' (str): The original input text.
            - 'x' (torch.Tensor): Tokenized tensor representation of the input text.
            - 'x_lengths' (torch.Tensor): Length of the tokenized input.
            - 'x_phones' (str): Human-readable phoneme sequence from the tokens.
    """
    x = torch.tensor(intersperse(matcha_text_to_sequence(text, ['basic_cleaners'])[0], 0),dtype=torch.long,device=device)[None, :]
    x_lengths = torch.tensor([x.shape[-1]],dtype=torch.long, device=device)
    x_phones = matcha_sequence_to_text(x.squeeze(0).tolist())
    return {
        'x_orig': text,
        'x': x,
        'x_lengths': x_lengths,
        'x_phones': x_phones
    }

@torch.inference_mode()
def infer_matcha(model_cfg: Dict,text: Union[str, torch.Tensor],device: torch.device) -> torch.Tensor:
    """
    Performs inference using the Matcha-TTS model on the given text input.

    Args:
        model_cfg (Dict): Configuration dictionary containing the model checkpoint path under `checkpoint`.
        text (Union[str, torch.Tensor]): The input text to synthesize (can be raw text or pre-tokenized tensor).
        device (torch.device): The device to run inference on (CPU or CUDA).

    Returns:
        Dict[str, Union[torch.Tensor, str, datetime.datetime]]: A dictionary containing:
            - 'audio' (torch.Tensor): The generated waveform or mel-spectrogram (depending on model).
            - 'start_t' (datetime.datetime): Timestamp when inference started.
            - 'x_orig' (str): The original input text.
            - 'x' (torch.Tensor): Tokenized tensor input.
            - 'x_lengths' (torch.Tensor): Sequence length tensor.
            - 'x_phones' (str): Readable phoneme string of the input.
    """
    # Load the model
    checkpoint_path = os.path.join(model_cfg.savedir,model_cfg.checkpoint)
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device).eval().to(device)
    text_processed = process_text(text, device)
    start_t = datetime.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=model_cfg.n_timesteps,
        temperature=model_cfg.temperature,
        spks=model_cfg.spks,
        length_scale=model_cfg.length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

# === Tacotron2 model inference ===
@torch.inference_mode()
def infer_tacotron2(cfg: Dict, text: Union[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Runs inference using a Tacotron2 model to generate a mel-spectrogram from input text.

    Args:
        cfg (Dict): Dictionary containing the configuration.
        text (Union[str, torch.Tensor]): Input text as a string or pre-tokenized tensor.
        device (torch.device): The device to run inference on (e.g., torch.device("cuda")).

    Returns:
        torch.Tensor: The generated mel-spectrogram tensor.
    """
    hparams = create_hparams()
    model_cfg = cfg.model_tacotron2
    hparams.sampling_rate = cfg.sampling_rate
    model = load_model(hparams)
    checkpoint_path = os.path.join(model_cfg.savedir, model_cfg.checkpoint)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'], strict=False)
    model.eval()
    model = model.to(device)
    sequence = np.array(tacotron_text_to_sequence(text, ['basic_cleaners']))[None, :]  
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)) 
    sequence = sequence.to(device).long()
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    return mel_outputs_postnet

# === Vocoder inference ===
@torch.inference_mode()
def vocode_hifigan(vocoder_cfg: Dict, mel: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Synthesizes waveform from a mel-spectrogram using a pretrained HiFi-GAN vocoder.

    Args:
        vocoder_cfg (Dict): Configuration dictionary with `source` and `savedir` for loading HiFi-GAN.
        mel (torch.Tensor): Mel-spectrogram of shape (n_mels, time).
        device (torch.device): Device to run the vocoder on.

    Returns:
        torch.Tensor: Synthesized waveform as a 1D tensor.
    """
    mel = mel.to(device).float()
    
    # Load HiFi-GAN vocoder
    voc = SBHifiGAN.from_hparams(
        source=vocoder_cfg['source'],
        savedir=vocoder_cfg['savedir'],
        run_opts={"device": device}
    )
    voc.eval()
    voc = voc.to(device)
    # Generate waveform
    waveform = voc.decode_batch(mel).squeeze()
    return waveform

@torch.inference_mode()
def vocode_bigvgan(vocoder_cfg: Dict,mel: torch.Tensor ,device: torch.device) -> torch.Tensor:
    """
    Synthesizes waveform from a mel-spectrogram using a pretrained BigVGAN vocoder.

    Args:
        vocoder_cfg (Dict): Configuration with a `source` key specifying the model location.
        mel (torch.Tensor): Mel-spectrogram tensor of shape (n_mels, time).
        device (torch.device): Device to run the vocoder on (e.g., torch.device("cuda")).

    Returns:
        torch.Tensor: The generated waveform tensor.
    """
    mel = mel.to(device).float()  # Ensure mel is on the correct device
    bigvgan_path = os.path.join(vocoder_cfg.savedir, vocoder_cfg.checkpoint)
    voc = BigVGAN.from_pretrained(vocoder_cfg.checkpoint,cache_dir = vocoder_cfg.savedir, use_cuda_kernel=False).to(device)
    voc.remove_weight_norm()
    voc.eval() 
    return voc(mel).squeeze()

@torch.inference_mode()
def vocode_griffinlim(cfg: Dict,mel: torch.Tensor) -> torch.Tensor:
    """
    Converts a mel-spectrogram to waveform using the Griffin-Lim algorithm.

    Args:
        cfg (Dict): Configuration dictionary.
        mel (torch.Tensor): Mel-spectrogram of shape (n_mels, time).

    Returns:
        torch.Tensor: Reconstructed waveform as a 1D tensor.
    """
    griffinlim_cfg = cfg.vocoder_griffinlim
    mel = mel.cpu().detach().numpy()
    mel = np.exp(mel)
    stft = librosa.feature.inverse.mel_to_stft(mel, sr = cfg.sampling_rate, n_fft= griffinlim_cfg.n_fft)
    return librosa.griffinlim(stft, n_iter=griffinlim_cfg.n_iter ,hop_length = griffinlim_cfg.hop_length, win_length = griffinlim_cfg.win_length)

def vocode_ringformer(vocoder_cfg: Dict, mel: torch.Tensor, SPEAKER_ID: int, device: torch.device) -> torch.Tensor:
    """
    Synthesizes waveform from mel-spectrogram using Ringformer/VITS-style model.

    Args:
        vocoder_cfg (Dict): Configuration dictionary. Must contain 'hps' and 'checkpoint_path'.
        mel (torch.Tensor): Input mel-spectrogram tensor of shape [1, 80, T].
        SPEAKER_ID (int): ID of the speaker for multi-speaker synthesis.
        device (torch.device): Target device.

    Returns:
        torch.Tensor: Generated waveform as a 1D tensor.
    """
    config_path = os.path.join(vocoder_cfg['savedir'], vocoder_cfg['config'])
    hps = get_hparams_from_file(config_path)
    CHECKPOINT_PATH = os.path.join(vocoder_cfg['savedir'], vocoder_cfg['checkpoint_path'])

    if hps['model']['use_mel_posterior_encoder']:
        print("Using mel posterior encoder for VITS2")
        posterior_channels = 80
        hps['data']['use_mel_posterior_encoder'] = True
    else:
        print("Using lin posterior encoder for VITS1")
        posterior_channels = hps['data']['filter_length'] // 2 + 1
        hps['data']['use_mel_posterior_encoder'] = False

    # Build the model
    net_g = SynthesizerTrn(
        n_vocab=0,
        spec_channels=posterior_channels,
        segment_size=hps['train']['segment_size'] // hps['data']['hop_length'],
        n_speakers=hps['data']['n_speakers'],
        **hps['model']
    ).to(device)
    net_g.enc_p.emb = torch.nn.Embedding(178, 192)

    _ = net_g.eval()
    _ = load_checkpoint(CHECKPOINT_PATH, net_g, None)

    mel_tensor = mel.to(device).float()  # [1, 80, T]
    mel_lengths = torch.LongTensor([mel_tensor.shape[2]]).to(device)

    # Speaker embedding
    g = None

    # Synthesis
    with torch.no_grad():
        net_g.eval()
        z, _, _, y_mask = net_g.enc_q(mel_tensor, mel_lengths)
        audio, _, _ = net_g.dec(z * y_mask, g=g)
        audio = audio.squeeze().cpu()

    return audio



# === Unified Runner ===
def run_inference(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using config:\n", OmegaConf.to_yaml(cfg))
    text = cfg.inference.text
    text = HebrewToEnglish(text)  # Convert Hebrew text to English sounds
    # Model inference
    if cfg.model.type == "matcha":
        output = infer_matcha(cfg.model_matcha, text, device)
        mel = output['mel']  # Extract mel-spectrogram from the output
    elif cfg.model.type == "tacotron2":
        mel = infer_tacotron2(cfg, text, device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")

    # Vocoder
    if cfg.vocoder.type == "hifigan":
        audio = vocode_hifigan(cfg.vocoder_hifigan, mel, device)
    elif cfg.vocoder.type == "bigvgan":
        audio = vocode_bigvgan(cfg.vocoder_bigvgan, mel, device)
    elif cfg.vocoder.type == "griffinlim":
        audio = vocode_griffinlim(cfg, mel)
    elif cfg.vocoder.type == "ringformer":
        audio = vocode_ringformer(cfg.vocoder_ringformer, mel, 0,device)
    else:
        raise ValueError(f"Unknown vocoder type: {cfg}")

    # Save audio
    output_file = Path(cfg.inference.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_file, audio.cpu().numpy(), cfg.sampling_rate)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    if "--text" in sys.argv:
        # ==== Classic CLI Mode ====
        parser = argparse.ArgumentParser(
            description="Inference script for TTS models (Tacotron2 / Matcha) and vocoders (HiFi-GAN, BigVGAN, etc.)",
            epilog="""Example:
            python inference.py \\
                --text "שלום עולם" \\
                --model tacotron2 \\
                --vocoder hifigan \\
                --checkpoint /gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/Pytorch/checkpoints/matcha_tts/logs/train/ljspeech/runs/2025-06-02_14-11-39/checkpoints/checkpoint_epoch=2679.ckpt \\
                --output-file outputs/generated.wav
            """
        )
        parser.add_argument("--text", type=str, required=True,help="Input text to synthesize (Hebrew supported via phonetic mapping).")

        parser.add_argument("--model", type=str, choices=["tacotron2", "matcha"], default="matcha", required=True,help="TTS model to use. Options: 'tacotron2', 'matcha'. Default: 'matcha'.")

        parser.add_argument("--vocoder", type=str, choices=["hifigan", "bigvgan", "griffinlim", "ringformer"], default="hifigan", required=True,help="Vocoder to convert mel-spectrogram to waveform. Default: 'hifigan'.")

        parser.add_argument("--checkpoint", type=str, required=True,help="Path to the model/vocoder checkpoint file.")

        parser.add_argument("--output_file", type=str, default="output.wav",help="Path where the synthesized waveform will be saved.")

        parser.add_argument("--sampling_rate", type=int, default=22050,help="Sampling rate for output audio. Default: 22050 Hz.")
        args = parser.parse_args()

        # Manually build config from argparse
        cfg = OmegaConf.create({
            "inference": {
                "text": args.text,
                "output_file": args.output_file
            },
            "model": {
                "type": args.model
            },
            "vocoder": {
                "type": args.vocoder
            },
            "sampling_rate": args.sampling_rate,
            # Set up basic checkpointing config so inference works
            "model_tacotron2": {
                "savedir": os.path.dirname(args.checkpoint),
                "checkpoint": os.path.basename(args.checkpoint)
            },
            "vocoder_hifigan": {
                "source": None,
                "savedir": os.path.dirname(args.checkpoint),
                "checkpoint": os.path.basename(args.checkpoint)
            },
            "vocoder_bigvgan": {
                "savedir": os.path.dirname(args.checkpoint),
                "checkpoint": os.path.basename(args.checkpoint)
            },
            "vocoder_ringformer": {
                "savedir": os.path.dirname(args.checkpoint),
                "checkpoint_path": os.path.basename(args.checkpoint),
                "config": "config.json"  # or .yaml depending on your setup
            },
            "vocoder_griffinlim": {
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024
            }
        })

        run_inference(cfg)

    else:
        # ==== Hydra config mode ====
        @hydra.main(config_path="config", config_name="config.yaml")
        def hydra_main(cfg: DictConfig):
            run_inference(cfg)
        hydra_main()
