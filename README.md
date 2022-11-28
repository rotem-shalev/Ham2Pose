# Ham2Pose
Official implementation for "Ham2Pose: Animating Sign Language Notation into Pose Sequences".
The first method for animating HamNoSys into pose sequences.

<p align="center">
  <img src="results_example/results.gif">
</p>

## Getting Started

1. Create a virtual environment 
```bash
$ conda create --name Ham2Pose python=3.7
$ conda activate Ham2Pose
$ pip3 install requirements.txt
```

2. Prepare new data: To train the model using data that isn't part of our dataset, download the videos and use
 [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to extract body, face, and hands keypoints from them (137 in total).
 
  
## Train

To train a new model using the default configurations run the following command:
```bash
$ python train.py
```
To change specific arguments, pass them as explained in the `args.py` file, e.g. to change the number of
 gpus to 4 and the batch size to 32 use:
```bash
$ python train.py --num_gpus 4 --batch_size 32
```
To pass a complete yaml configuration file use:
```bash
$ python train.py --config_file configs/config.yaml
```

## Test

To test an existing model with the default configuration use:
```bash
$ python test.py
```

The default configuration will test our supplied pretrained model "Ham2Pose". To train and test a different model, either change the `model_name` in the configuration, or delete the existing checkpoint from the `models` directory.

To change other arguments use one of the options mentioned under Train.

## Citation

If you find this research useful, please cite the following:
```
@misc{https://doi.org/10.48550/arxiv.2211.13613,
  doi = {10.48550/ARXIV.2211.13613},
  url = {https://arxiv.org/abs/2211.13613},
  author = {Shalev-Arkushin, Rotem and Moryossef, Amit and Fried, Ohad},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Ham2Pose: Animating Sign Language Notation into Pose Sequences},  
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution Share Alike 4.0 International}
}
```
