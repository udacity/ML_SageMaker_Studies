# Machine Learning, Deployment Case Studies with AWS SageMaker

This repository contains code and associated files for deploying ML models using AWS SageMaker. This repository consists of a number of tutorial notebooks for various case studies, code exercises, and project files that will be to illustrate parts of the ML workflow and give you practice deploying a variety of ML algorithms.

### Tutorials
* [Population Segmentation](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Population_Segmentation): Learn how to build and deploy unsupervised models in SageMaker. In this example, you'll cluster US Census data; reducing the dimensionality of data using PCA and the clustering the resulting, top components with k-means.

### Project

[Plagiarism Detector](https://github.com/udacity/ML_SageMaker_Studies/tree/master/Project_Plagiarism_Detection): Build an end-to-end plagiarism classification model. Apply your skills to clean data, extract meaningful features, and deploy a plagiarism classifier in SageMaker.

---

## Setup Instructions

The notebooks provided in this repository are intended to be executed using Amazon's SageMaker platform. The following is a brief set of instructions on setting up a managed notebook instance using SageMaker, from which the notebooks can be completed and run.

### Log in to the AWS console and create a notebook instance

Log in to the [AWS console](https://console.aws.amazon.com) and go to the SageMaker dashboard. Click on 'Create notebook instance'.
* The notebook name can be anything and using ml.t2.medium is a good idea as it is covered under the free tier. 
* For the role, creating a new role works fine. Using the default options is also okay. 
* It's important to note that you need the notebook instance to have access to S3 resources, which it does by default. In particular, any S3 bucket or object, with sagemaker in the name, is available to the notebook.
* Use the option to **git clone** the project repository into the notebook instance by pasting `https://github.com/udacity/ML_SageMaker_Studies.git`

### Open and run the notebook of your choice

Now that the repository has been cloned into the notebook instance you may navigate to any of the notebooks that you wish to complete or execute and work with them. Additional instructions are contained in their respective notebooks.
