# Astroformer

This repository contains the official implementation of Astroformer, an ICLR Workshop 2023 paper. This model is aimed at detection tasks in the low-data regimes and achieves SoTA results on CIFAR-100, Tiny Imagenet, and science tasks like Galaxy10 DECals, and competetive performance on CIFAR-10 _without any additional labelled or unlabelled data_.

_**Accompanying paper: [Astroformer: More Data Might not be all you need for Classification](https://arxiv.org/abs/2304.05350)**_ [![arXiv](https://img.shields.io/badge/paper-arXiv:2304.05350-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2304.05350)

## Code Overview

The most important code is in `astroformer.py`. We trained Astroformers using the `timm` framework, which we copied from [here](https://github.com/huggingface/pytorch-image-models).

Inside `pytorch-image-models`, we have made the following modifications. (Though one could look at the diff, we think it is convenient to summarize them here.)

- added `timm/models/astroformer.py`
- modified `timm/models/__init__.py`

## Training

If you had a node with 8 GPUs, you could train a Astroformer 5 as follows (these are exactly the settings we used for Galaxy10 DECals as well):

```sh
sh distributed_train.sh 8 [/path/to/dataset] 
    --train-split [your_train_dir] 
    --val-split [your_val_dir] 
    --model astroformer_5
    --num-classes 10
    --img-size 256
    --in-chans 3
    --input-size 3 256 256
    --batch-size 256
    --grad-accum-steps 1
    --opt adamw
    --sched cosine
    --lr-base 2e-5
    --lr-cycle-decay 1e-2
    --lr-k-decay 1
    --warmup-lr 1e-5
    --epochs 300
    --warmup-epochs 5
    --mixup 0.8
    --smoothing 0.1
    --drop 0.1
    --save-images
    --amp
    --amp-impl apex
    --output result_ours/astroformer_5_galaxy10
    --log-wandb
```

You could simply use the same script with the other Astrofromer models: `astroformer_0`, `astroformer_1`, `astroformer_2`, `astroformer_3`, `astroformer_4`, and `astroformer_5` to train those variants as well.

## Main Results

### CIFAR-100

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 87.65          | 31.36 | 161.95 |
| Astroformer-4| 93.36          | 60.54 | 271.68 |
| Astroformer-5| 89.38          | 115.97| 655.34 |

### CIFAR-10

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 99.12          | 31.36 | 161.75 |
| Astroformer-4| 98.93          | 60.54 | 271.54 |
| Astroformer-5| 93.23          | 115.97| 655.04 |

### Tiny Imagenet

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 86.86          | 24.84 | 150.39 |
| Astroformer-4| 91.12          | 40.38 | 242.58 |
| Astroformer-5| 92.98          | 89.88 | 595.55 |

### Galaxy10 DECals

| Model Name   | Top-1 Accuracy | FLOPs | Params |
|--------------|----------------|-------|--------|
| Astroformer-3| 92.39          | 31.36 | 161.75 |
| Astroformer-4| 94.86          | 60.54 | 271.54 |
| Astroformer-5| 94.81          | 105.9 | 681.25 |

## Citation

If you use this work, please cite the following paper:

BibTeX:

```bibtex
@article{dagli2023astroformer,
  title={Astroformer: More Data Might Not be All You Need for Classification},
  author={Dagli, Rishit},
  journal={arXiv preprint arXiv:2304.05350},
  year={2023}
}
```

MLA:

```
Dagli, Rishit. "Astroformer: More Data Might Not be All You Need for Classification." arXiv preprint arXiv:2304.05350 (2023).
```


## Credits

The code is heavily adapted from [timm](https://github.com/huggingface/pytorch-image-models).
