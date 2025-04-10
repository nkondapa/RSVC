# Representational Similarity via Interpretable Visual Concepts (RSVC)

[![OpenReview](https://img.shields.io/badge/OpenReview-a61717)](https://openreview.net/forum?id=ih3BJmIZbC)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)]([https://arxiv.org/abs/2310.00031](https://arxiv.org/pdf/2503.15699))

### Setup
```sh

conda create -n "RSVC" python=3.8.18
conda activate RSVC
bash setup.sh

```

### Downloading Datasets
1) ImageNet: You can download imagenet with ```bash download_imagenet.sh```.
2) NABirds: Please visit https://dl.allaboutbirds.org/nabirds to request access and download the dataset.

The datasets should be placed (or symlinked) in the `data/` directory.
The structure should look like this:
```
data/
│-- imagenet/
│   │-- train/
│   │-- val/
│   │-- ILSVRC2012_devkit_t12/
│   │-- ...
|-- nabirds/
    |-- images/
    |-- parts/
    |-- ...
```

## Usage
### Toy Concept Comparison
To run the toy concept comparison experiment, make sure the nabirds dataset is downloaded.
You can download the custom trained models from [M_ps](https://drive.google.com/drive/folders/1oG6uHMPahBYVc-AtqVYSZyWAvq8u74zw?usp=drive_link) and
[M_nc](https://drive.google.com/drive/folders/1LPXUh_Q3J9CCAg__o-T3RAcoVAAQ8ZRe?usp=drive_link).

```bash toy_concept_experiment.sh```

### RN18 vs. RN50
To run the resnet size comparison make sure the imagenet dataset is downloaded and placed in the correct location.
These models should automatically download from the timm library.

```bash compare_rn18_rn50.sh```

## TODO
- [ ] Add qualitative visualization code to resnet comparisons
- [ ] Add remaining comparison experiments


# Citation
```
@inproceedings{kondapaneni2025representational,
  title={Representational Similarity via Interpretable Visual Concepts},
  author={Kondapaneni, Neehar and Mac Aodha, Oisin and Perona, Pietro},
  journal={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
