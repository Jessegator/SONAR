# SONAR: A Synthetic AI-Audio Detection Framework and Benchmark
An official implementation of "SONAR: A Synthetic AI-Audio Detection Framework and Benchmark"

## Enviroment

- ``conda env export > environment.yml``
- ``conda activate sonar``

## Datasets

Please download the following datasets and extract them in ``/data/``  or change the database path correspondingly in the code.

- [Wavefake](https://zenodo.org/records/5642694) 
- [LibriSeVoc](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts)
- [In-the-wild](https://deepfake-total.com/in_the_wild)
- [LJ-Speech](https://keithito.com/LJ-Speech-Dataset/)
- [SONAR](https://drive.google.com/drive/folders/1kSqjuHiElNigCvGxD6sVKiyVaXA3xO5A?usp=sharing)

The structure should look like

```
data
├── LJSpeech-1.1
│ 	├── wavs
│	├── metadata.csv
│ 	└── README
├── wavefake
│ 	├── ljspeech_full_band_melgan
│	├── ljspeech_hifiGAN
│	├── ...
│ 	└── ljspeech_waveglow
├── LibriSeVoc
│ 	├── diffwave
│	├── gt
│	├── ...
│ 	└── wavernn
├── in_the_wild
│ 	├── 0.wav
│	├── ...
│	├── 31778.wav
│ 	└── meta.csv
```



## Usage examples

To train traditional models, please run ``main_tm.py``

Arguments

- ``--config``: config files for different models.

- Train AASIST on wavefake

  ```
  python main_tm.py --config ./config/AASIST.conf 
  ```

- Evaluation (modify the ``model_path`` in corresponding config files.)

  ```
  python main_tm.py --config ./config/AASIST.conf --eval
  ```

To fine-tune foundation models, please run ``main_fm.py``

- Fine-tune Wave2Vec2BERT

  ```
  python main_fm.py --model wave2vec2bert
  ```

  

## Acknowledgements

This repository is built on top of the following open source projects.

- [AASIST](https://github.com/clovaai/aasist)
- [wavefake](https://github.com/RUB-SysSec/WaveFake/tree/main)
