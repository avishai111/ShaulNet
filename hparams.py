from dataclasses import dataclass
from text import symbols

@dataclass
class HParams:
    # Experiment Parameters
    epochs: int = 1000
    iters_per_checkpoint: int = 1000
    seed: int = 1235
    dynamic_loss_scaling: bool = True
    fp16_run: bool = False
    distributed_run: bool = False
    dist_backend: str = "nccl"
    dist_url: str = "tcp://localhost:54321"
    cudnn_enabled: bool = True
    cudnn_benchmark: bool = False
    ignore_layers: list = ('embedding.weight',)

    # Data Parameters
    load_mel_from_disk: bool = True
    training_files: str = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/train_with_80percent_dev_weighted.txt'
    validation_files: str = '/gpfs0/bgu-benshimo/users/wavishay/VallE-Heb/TTS2/tacotron2/dev_20percent_list_weighted.txt'
    text_cleaners: list = ('basic_cleaners',)

    # Audio Parameters
    max_wav_value: float = 32768.0
    sampling_rate: int = 22050
    filter_length: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mel_channels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000.0

    # Model Parameters
    n_symbols: int = len(symbols)
    symbols_embedding_dim: int = 512

    # Encoder parameters
    encoder_kernel_size: int = 5
    encoder_n_convolutions: int = 3
    encoder_embedding_dim: int = 512

    # Decoder parameters
    n_frames_per_step: int = 1  # currently only 1 is supported
    decoder_rnn_dim: int = 1024
    prenet_dim: int = 256
    max_decoder_steps: int = 1000
    gate_threshold: float = 0.5
    p_attention_dropout: float = 0.1
    p_decoder_dropout: float = 0.1

    # Attention parameters
    attention_rnn_dim: int = 1024
    attention_dim: int = 128

    # Location Layer parameters
    attention_location_n_filters: int = 32
    attention_location_kernel_size: int = 31

    # Mel-post processing network parameters
    postnet_embedding_dim: int = 512
    postnet_kernel_size: int = 5
    postnet_n_convolutions: int = 5

    # Optimization Hyperparameters
    use_saved_learning_rate: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_thresh: float = 1.0
    batch_size: int = 32 #32
    mask_padding: bool = True  # set model's padded outputs to padded values


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""
    hparams = HParams()

    # Optional: parse command-line string input to update hyperparameters
    if hparams_string:
        for param in hparams_string.split(','):
            name, value = param.split('=')
            setattr(hparams, name.strip(), eval(value.strip()))

    if verbose:
        print(f"Final parsed hparams: {hparams}")

    return hparams
