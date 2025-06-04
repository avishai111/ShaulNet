import torch
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
from tracatron2.text import text_to_sequence as tacotron_text_to_sequence

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
def process_text(text: str, device: torch.device) -> Dict[str, torch.Tensor | str]:
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
    x = torch.tensor(intersperse(matcha_text_to_sequence(text, ['basic_cleaners'])[0], 0),dtype=torch.long, device=device)[None]
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
    ## Number of ODE Solver steps
    n_timesteps = 10
    ## Changes to the speaking rate
    length_scale = 1
    ## Sampling temperature
    temperature = 0.1
    # Load the model
    model = MatchaTTS.load_from_checkpoint(model_cfg.checkpoint, map_location=device).eval().to(device)
    text_processed = process_text(text, device)
    start_t = datetime.datetime.now()
    output = model.synthesise(
        text_processed['x'], 
        text_processed['x_lengths'],
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=None,
        length_scale=length_scale
    )
    # merge everything to one dict    
    output.update({'start_t': start_t, **text_processed})
    return output

# === Tacotron2 model inference ===
@torch.inference_mode()
def infer_tacotron2(model_cfg: Dict, text: Union[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    """
    Runs inference using a Tacotron2 model to generate a mel-spectrogram from input text.

    Args:
        model_cfg (Dict): Dictionary containing model configuration. Must include a 'checkpoint' path.
        text (Union[str, torch.Tensor]): Input text as a string or pre-tokenized tensor.
        device (torch.device): The device to run inference on (e.g., torch.device("cuda")).

    Returns:
        torch.Tensor: The generated mel-spectrogram tensor.
    """
    model = Tacotron2.from_hparams(source=model_cfg.source, savedir=model_cfg.savedir,
                                   run_opts={"device": device})
    seq = model.encode_text(text)
    mel = model(seq).squeeze(0).permute(1, 0)  # [T, 80] → [80, T]
    return mel

# === Vocoder inference ===
@torch.inference_mode()
def vocode_hifigan(vocoder_cfg: Dict, mel: torch.Tensor,device: torch.device) -> torch.Tensor:
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
    waveform = voc.decode_batch(mel.unsqueeze(0)).squeeze()
    return waveform
@torch.inference_mode()
def vocode_bigvgan(vocoder_cfg: Dict,mel: torch.Tensor,device: torch.device) -> torch.Tensor:
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
    voc = BigVGAN.from_pretrained(vocoder_cfg.source, use_cuda_kernel=False).to(device).eval()
    voc.remove_weight_norm()
    voc = voc.eval().to(device) 
    return voc(mel.unsqueeze(0)).squeeze()

@torch.inference_mode()
def vocode_griffinlim(vocoder_cfg: Dict,mel: torch.Tensor) -> torch.Tensor:
    """
    Converts a mel-spectrogram to waveform using the Griffin-Lim algorithm.

    Args:
        vocoder_cfg (Dict): Configuration dictionary with:
            - 'sampling_rate': Sampling rate of the audio.
            - 'hop_length': Hop length for STFT.
            - 'win_length': Window length for STFT.
        mel (torch.Tensor): Mel-spectrogram of shape (n_mels, time).

    Returns:
        torch.Tensor: Reconstructed waveform as a 1D tensor.
    """
    mel = mel.cpu().detach().numpy()
    mel = np.exp(mel)
    stft = librosa.feature.inverse.mel_to_stft(mel, sr=vocoder_cfg.sampling_rate, n_fft=1024)
    return librosa.griffinlim(stft, n_iter=100,hop_length = vocoder_cfg.hop_length, win_length=vocoder_cfg.win_length)


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
    hps = vocoder_cfg['hps']
    CHECKPOINT_PATH = vocoder_cfg['checkpoint_path']

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
    _ = util.load_checkpoint(CHECKPOINT_PATH, net_g, None)

    mel_tensor = mel.to(device).float()  # [1, 80, T]
    mel_lengths = torch.LongTensor([mel_tensor.shape[2]]).to(device)

    # Speaker embedding
    if hps['data']['n_speakers'] > 0:
        sid = torch.LongTensor([SPEAKER_ID]).to(device)
        g = net_g.emb_g(sid).unsqueeze(-1)
    else:
        g = None

    # Synthesis
    with torch.no_grad():
        z, _, _, y_mask = net_g.enc_q(mel_tensor, mel_lengths)
        audio, _, _ = net_g.dec(z * y_mask, g=g)
        audio = audio.squeeze().cpu()

    return audio



# === Main  ===
@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using config:\n", OmegaConf.to_yaml(cfg))
    text = cfg.inference.text
    
    text = HebrewToEnglish(text)  # Convert Hebrew text to English sounds

    # Model inference
    if cfg.model.type == "matcha":
        mel = infer_matcha(cfg.model, text, device)
    elif cfg.model.type == "tacotron2":
        mel = infer_tacotron2(cfg.model, text, device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")

    # Vocoder
    if cfg.vocoder.type == "hifigan":
        audio = vocode_hifigan(cfg.vocoder, mel, cfg.model.type, device)
    elif cfg.vocoder.type == "bigvgan":
        audio = vocode_bigvgan(cfg.vocoder, mel, device)
    elif cfg.vocoder.type == "griffinlim":
        audio = vocode_griffinlim(cfg.vocoder, mel)
    elif cfg.vocoder.type == "ringformer":
        audio = vocode_ringformer(cfg.vocoder, mel, 0,device)
    else:
        raise ValueError(f"Unknown vocoder type: {cfg.vocoder.type}")

    # Save audio
    output_path = Path(cfg.inference.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio.cpu().numpy(), cfg.vocoder.sampling_rate)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()