# Udacity's Data Scientist Nanodegree Capstone Project & Deep Learning Nanodegree Project: Dog Breed Classifier

## Project Overview

This project is a real-world application of the Convolutional Neural Networks (CNNs) of the Deep Learning (DL) field. The aim of the project is to build a model to classify the image data. Given an image, the model is able to distinguish whether being a dog image and to identify the breed of the dog if it is the case. In addition to building a model from scratch, the project also uses the Transfer Learning technique so that the model pre-trained could be reused across different tasks, saving the resource of calculation while still achieving a reasonable accuracy.

## Dependencies 

Keras

OpenCV

Matplotlib

Numpy

tqdm

Pytorch

PIL

os

A more detailed description of the libraries and dependencies is available in requirement.txt

## File Descriptions 

dog_app_finished.ipynb: A Jupyter Notebook file describing the product notebook of the training and validation process

dog_app_original.ipynb: A Jupyter Notebook file containing the raw source code from the Udacity Project

report.html: An example file of the final output

requirement.txt: File describing the environment and the essential libraries

images\*: A folder containing a few important images for testing the model

saved_models\model_scratch.pt: A model built from scratch for the project

saved_models\model_transfer.pt: A resnet50 model trained for the dog breed classification 

## How to Use

It is recommended to follow the instruction of the Jupyter Notebook to train and test your model. 

The steps could roughly be summarized as the following:

0. Import Datasets
1. Detect Humans
2. Detect Dogs
3. Create a CNN to Classify Dog Breeds (from Scratch)
4. Create a CNN to Classify Dog Breeds (using Transfer Learning)
5. Write your Algorithm
6. Test Your Algorithm

You can also use the pt models as a starting point to work on your own task specific project. 

## To be improved 

* The accuracy of the model could be improved with trying to optimize the hyperparameters and the grid search strategies.

* The images for training are downloaded from the internet (links are available in the Notebook file). However, the model might potentially be improved with more images with a wider range of dog breeds and more images related to each category. 

* Another solution to improve the performance is to add random noise or rotate the images, so that the model might improve without needing more labeled images.  

* It is currently hard for the model to distinguish between dogs with similar colors but different sizes. An alternative might be helpful if it could pick up more subtle differences. 
