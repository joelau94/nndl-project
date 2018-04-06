# Reimplementation of Deeplab
## Requirement
python 3, pytorch, opencv-python

## How to run
Put data in root directory. ([PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/))

###For training
```
python train.py
```
###For testing
```
python test.py <model_file> <output_file>
```
## About this implementation
As I'm an NLP person, I'm not familiar with the processing of image data. So I used a lot of code snippets from [this implementation](https://github.com/speedinghzl/Pytorch-Deeplab) to do data processing and loading. But the core ```models.py``` was written on my own, though after reading their codes.

I was encountered with two problems: 1. When I use gpu, I was able to run the model (and verify that my implementation is correct). But there would be gpu issue comming up during backward pass ([known issue](https://github.com/SeanNaren/deepspeech.pytorch/issues/32)). 2. When I use cpu, my memory was not enough.

So what I did was to verify my implementation of the core model is correct on gpu. I wasn't able to complete the whole training and testing.
