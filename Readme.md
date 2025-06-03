# 🤖 ShaulNet - Robo-Shaul-TTS
---

## About the Robo-Shaul Challenge

The **Robo-Shaul Challenge** was a competition held from February to May 2023, aiming to develop the first open-source Hebrew text-to-speech (TTS) engine that replicates the voice of journalist and podcast host Shaul Amsterdamski. The initiative was documented in three podcast episodes, detailing the process and results.

As part of the challenge, a Hebrew single-speaker dataset named **SASPEECH** was created, consisting of approximately 30 hours of Amsterdamski's recordings. This dataset, along with a benchmark TTS system, was presented at INTERSPEECH 2023.

For more details, visit the [official Robo-Shaul website](https://www.roboshaul.com/).

---

**ShaulNet** is our solution to the [Robo-Shaul Challenge](https://www.roboshaul.com/): a flexible, multilingual Text-to-Speech (TTS) system designed to synthesize natural-sounding Hebrew speech from raw text using a modular and configurable architecture.

This repository integrates and builds upon several outstanding open-source projects, including:

* [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
* [RingFormer](https://github.com/seongho608/RingFormer)
* [Tacotron2](https://github.com/NVIDIA/tacotron2)
* [HiFi-GAN](https://github.com/jik876/hifi-gan)
* [BigVGAN](https://github.com/NVIDIA/BigVGAN)
* [Griffin-Lim](https://pytorch.org/audio/main/generated/torchaudio.transforms.GriffinLim.html)
* [WaveGlow](https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/)

---

## 🚀 Features

* 🎙️ High-quality text-to-speech synthesis
* 🌐 Full Hebrew language support, including diacritics and phonetic normalization
* 🧠 Built on state-of-the-art models and vocoders
* 🧪 Modular architecture for research, experimentation, and production

---

## 📝 Basic Usage

```bash
# Run TTS using the config file (choose model/vocoder inside it)
python inference.py
````

Edit the `conf/config.yaml` file to customize the input text, model, vocoder, and output:

```yaml
inference:
  text: "מה קורה, שאול?"
  output_path: "outputs/shaul.wav"
```

---

## 🛠️ Project Structure

```
ShaulNet/
│
├── config/                             # YAML configuration files for Hydra (model/vocoder selection)
├── data/                               # Raw or processed datasets (e.g., text inputs or training data)
├── Data_Creation_scripts/              # Scripts for text preprocessing (e.g., Hebrew-to-English conversion)
├── download_pretrained_models_scripts/ # Scripts to download pretrained models (e.g., Tacotron2, WaveGlow)
├── filelists/                          # File lists for model training (e.g., train.txt, val.txt)
├── Matcha_TTS/                         # Matcha-TTS model codebase (training, synthesis, utils)
├── Plotting/                           # Utility scripts for visualizing mel-spectrograms and alignments
├── Preprocess/                         # Data preprocessing pipeline (cleaners, format converters, etc.)
├── pretrained_models/                  # Folder containing downloaded/pretrained TTS and vocoder models
├── RingFormer/                         # Optional RingFormer model implementation (if enabled)
├── tracatron2/                         # Tacotron2 model code (architecture, training, inference)
├── .gitignore                          # Git ignore rules (e.g., checkpoints, logs, outputs)
├── LICENSE                             # Open-source license (e.g., MIT)
├── inference.py                        # Main script to perform TTS inference via Hydra-configurable pipeline
└── README.md                           # This file
```

---

## 🙏 Acknowledgments

* This project is based in part on the **English-to-Hebrew preprocessing** logic from [maxmelichov/Text-To-speech](https://github.com/maxmelichov/Text-To-speech).
* We would like to thank the developers of all the referenced projects above for their open-source contributions to the speech synthesis community.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📞 Contact

For questions, issues, or collaboration ideas:

📧 [Avishai Weizman](mailto:Avishai11900@gmail.com)

🔗 [LinkedIn](https://www.linkedin.com/in/avishai-weizman/)

🐙 [GitHub](https://github.com/avishai111)


```

---


