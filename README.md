# A2: Abstraction-based verification

Due: April 10

## Requirements
Make sure Python3.6 or above is installed

To install Pytorch
```
pip3 install torch torchvision
```

To run the program
```
python3 feed-forward-nn.py
```

## Goals

This assignment is designed to help you understand abstraction-based techniques for verifying neural networks. Specifically, we will implement the simplest abstract domain discussed in class, the intervals abstraction.

## The code

The code supplied is *exactly* the same one used in A1.

## Your job

Your job is to, given an image I with label L, verify that changing the brightness slightly of any pixel in I still results in the label L.

Following our notation from class, this is the property we want for an image I with label L and some constant epsilon:
```
{|x - I| <= epsilon}
r <- f(x)
{argmax_i r_i = L}
```
Notice that we take the largest index of the size 10 output vector. (We're working with MNIST again)

You will verify the above property using interval abstraction.
In other words, every pixel of the input image is now an interval, instead of a single real number, and therefore every node in the network also spits out an interval of values.
Consult the notes for more details.

Here's how I would approach this assignment:
1. Create a data structure that represents a vector of intervals.
2. Implement all necessary operations over that data structure to mimic those performed by the neural network. For example, our network contains a ReLU layer (```self.relu(x)```), so you need a version of that function that applies ReLUs over intervals (```self.relu_interval(xi)```), etc.

What to hand in:
1. Unlike with A1, this should be really fast. So pick 3 values of epsilon, and run the verification on the whole test set of images, and submit the code that iterates through the dataset. You might not be able to prove a lot of images robust (remember, with abstraction-based techniques, sometimes the answer is *Don't know* when the abstraction is too imprecise). If you find that's the case, make epsilon smaller, or make it non-zero only for the first N pixels instead of all pixels.
2. If things work with the small network, try to add more layers, retrain and see what happens.
3. If you get really excited, try to implement the zonotopes domain, or a network with sigmoid/tanh instead of ReLUs.

## Hand in
Please send zip file to aws@cs.wisc.edu

This is an individual assignment, but feel free to discuss with colleagues.
