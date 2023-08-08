# Kernel Methods for Machine Learning

The goal of the project is to perform a classification task using kernel methods.
We want to predict whether a given DNA sequence is bounded or unbounded for a transcription factor of interest.  
Three data sets are given, corresponding to three different transcription factors. Each transcription factor describe a new classification task, then, three different predictive models should be implemented.

Learning algorithms are implemented "from scratch", only using optimization librairies.

Here is the structure of the code :
  -  ”models.py” implements learning algorithms which are SVM and KRR.
  -  ”kernels.py”  implements  kernels  which  are  Gaussian  Kernel,  Polynomial  Kernel  and  MismatchKernel (generalizing Spectrum Kernel).  
  -  ”tuning.py” is used for hyperparameters tuning.  For thispurpose, the annotated data is splitted into training and validation sets.  Measuring performanceson both sets allow us to detect underfitting and overfitting. 
  -  ”tools.py” implements functions for datareading and data augmentation.  
  -  ”main.py” implements the competition submission.

To reproduce the submission, the script ”start.sh” can be called from Python.
