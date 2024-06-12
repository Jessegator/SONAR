# SONAR: A Synthetic AI-Audio Detection Framework and Benchmark
An official implementation of the paper "SONAR: A Synthetic AI-Audio Detection Framework and Benchmark"

## Enviroment

- ``conda env export > environment.yml``

## Datasets

Please download the following datasets and extract them in ``/data/`` 

- [Wavefake](https://zenodo.org/records/5642694) 
- [LibriSeVoc](https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts)
- [In-the-wild](https://deepfake-total.com/in_the_wild)

The structure should look like

```
data
├── LJSpeech-1.1
│ 	├── wavs
│		├── metadata.csv
│ 	└── README
├── wavefake
│ 	├── ljspeech_full_band_melgan
│		├── ljspeech_hifiGAN
│		├── ...
│ 	└── ljspeech_waveglow
├── LibriSeVoc
│ 	├── diffwave
│		├── gt
│		├── ...
│ 	└── wavernn
├── in_the_wild
│ 	├── 0.wav
│		├── ...
│		├── 31778.wav
│ 	└── meta.csv
```



## Usage examples

Arguments

- ``--config``: config files for different models.
- ``--dataset``: Speficy training dataset
- ``--eval_dataset``: Specify evaluation dataset
- ``--finetune``: few-shot finetuning

- Train AASIST on wavefake

  ```
  python main.py --config ./config/AASIST.conf --dataset wavefake
  ```

- Evaluation (modify the ``model_path`` in corresponding config files.)

  ```
  python main.py --config ./config/AASIST.conf --eval --eval_dataset {dataset}
  ```

- Personalized finetuning (modify the ``model_path`` in corresponding config files,finetuning will resume on the specified checkpoint)

  ```
  python main.py --config ./config/AASIST.conf --finetune --resume
  ```



## Acknowledgements

This repository is built on top of open source project.

- [AASIST](https://github.com/clovaai/aasist)
- [wavefake](https://github.com/RUB-SysSec/WaveFake/tree/main)
