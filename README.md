# Voice-Conversion-GAN
Voice Conversion using Cycle GAN's (PyTorch Implementation)
(Note: Project Under Development)
## Usage

### Download Dataset

Download and unzip [VCC2016](https://datashare.is.ed.ac.uk/handle/10283/2211) dataset to designated directories.

```bash
$ python download.py --help
usage: download.py [-h] [--download_dir DOWNLOAD_DIR] [--data_dir DATA_DIR]
                   [--datasets DATASETS]

optional arguments:
  -h, --help            show this help message and exit
  --download_dir DOWNLOAD_DIR
                        Download directory for zipped data
  --data_dir DATA_DIR   Data directory for unzipped data
  --datasets DATASETS   Datasets available: vcc2016
```

For example, to download the datasets to ``download`` directory and extract to ``data`` directory:

```bash
$ python download.py --download_dir ./download --data_dir ./data --datasets vcc2016
```

### Preprocessing for Training
Preprocess voice data and stores it in numpy format in ../cache folder
```bash
$ python prepocess_training.py --help
usage: prepocess_training.py [-h] [--train_A_dir TRAIN_A_DIR] [--train_B_dir TRAIN_B_DIR]
```
For example, to train CycleGAN model for voice Conversion between ``SF1`` and ``TM1``:
```bash
$ python prepocess_training.py --train_A_dir ../data/vcc2016_training/SF1
                                --train_B_dir ../data/vcc2016_training/TM1
```                                



### Useful Tutorials

PyTorch Tutorial: https://github.com/yunjey/pytorch-tutorial

Gaussian Mixture Model Voice Conversion: https://r9y9.github.io/nnmnkwii/latest/nnmnkwii_gallery/notebooks/vc/01-GMM%20voice%20conversion%20(en).html

CycleGAN - VC:
http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/

Voice Conversion using Variational Auto-Encoder:
https://github.com/JeremyCCHsu/vae-npvc
