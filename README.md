# Abstraction-based verification of Neural Networks

## Goal ##

In this model, we first train a simple neural network on the MNIST training set to do digit classification. Next, we take the trained model and correctly classified image and attempt to verify robustness by checking intervals of pixels instead of exact pixel values. That is, instead of classifying an image, it can classify a range of images where each pixel is an interval instead of a specific value. This can tell us if changing any numbers of pixels by a small amount can change the classification. In this experiment, we saw that changing any numbers of pixels by 0.1%, only about 5% of images would not certainly be the correct classification. But by changing them by as much as 1%, over 60% of the images could become uncertain. Technically, this doesn't show that that those images _can_ be misclassified, it shows that the outputs (which are also intervals) overlap so the correct classification isn't guaranteed.

## Requirements
Make sure Python3.6 or above is installed

To install Pytorch
```
pip3 install torch torchvision
```

To train the network

```
python3 feed-forward-nn.py train
```

To run the program without training

```
python3 feed-forward-nn.py
```

## Optimization ##

- I originally wrote everything using a Range class and operations over that
    - Each NN forward pass took 0.24s (40 minutes total per epsilon)
- I tried changing to tuples instead of a class
    - Each NN forward pass took 0.22s (37 minutes total per epsilon)
- I changed to using tensor operations over lists of lower and upper bounds
    - Though this require more total operations, they were internally optimized and parallelized
    - Each NN forward pass took 0.013s (2.2 minutes total per epsilon)

## Output ##

- This is the output for the "test" version. You can re-train the model by using `python feed-forward-nn.py train`. Epsilon is the amount that each pixel can change up and down (ie `range = (x_i - epsilon, x_i + epsilon)` for all `i`)
- The "robust amount" means that the lower bound of the correct label's interval is greater than the upper bounds of all incorrect labels. This means that the maximum robustness is the normal accuracy (92.54%)

```
Timothys-MBP:A2-interval-abstraction tj$ python feed-forward-nn.py
Average time (1000 images): 0.013858854055404664
Average time (1000 images): 0.013556383848190308
Average time (1000 images): 0.014038251876831056
Average time (1000 images): 0.013524255990982056
Average time (1000 images): 0.01340032696723938
Average time (1000 images): 0.013654764890670777
Average time (1000 images): 0.013518330097198486
Average time (1000 images): 0.013457827091217041
Average time (1000 images): 0.013374399900436401
Robust amount for epsilon=0.001: 8989/10000
Average time (1000 images): 0.01341677713394165
Average time (1000 images): 0.013441201210021972
Average time (1000 images): 0.01343557596206665
Average time (1000 images): 0.013371975898742675
Average time (1000 images): 0.013478943824768067
Average time (1000 images): 0.01340594482421875
Average time (1000 images): 0.013447988033294677
Average time (1000 images): 0.01340087604522705
Average time (1000 images): 0.013499964952468873
Robust amount for epsilon=0.005: 6988/10000
Average time (1000 images): 0.013482644319534302
Average time (1000 images): 0.01337493896484375
Average time (1000 images): 0.013404582738876342
Average time (1000 images): 0.01344388508796692
Average time (1000 images): 0.01343005895614624
Average time (1000 images): 0.013392107009887695
Average time (1000 images): 0.013400310039520264
Average time (1000 images): 0.01338854193687439
Average time (1000 images): 0.013386988878250123
Robust amount for epsilon=0.01: 2849/10000
```

## Problem Statement

Given an image I with label L, verify that changing the brightness slightly of any pixel in I still results in the label L.

Following our notation from class, this is the property we want for an image I with label L and some constant epsilon:
```
{|x - I| <= epsilon}
r <- f(x)
{argmax_i r_i = L}
```
