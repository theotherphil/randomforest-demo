# randomforest-demo

Hacky WIP to visualise predictions from https://github.com/theotherphil/randomforest, similarly to http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html and https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/decisionForests_MSR_TR_2011_114.pdf.

Finds the connected components of non-white pixels in an input image, creates a training set using their colours and centres-of-mass, trains a forest using this dataset and classifies every point in the input image using this forest. The image showing predictions colours each pixel by performing a weighted sum of the input colours, with weights proportional to the probability of each class in the produced distribution.

# Example

## Input image
![Alt text](/data/four-class-spiral.png?raw=true "Input image")

## Centres of connected components
![Alt text](/data/centres.png?raw=true "Dataset")

## Predictions 
![Alt text](/data/classification.png?raw=true "Predictions")

(200 trees, 500 candidate classifiers per node, tree depth 9, hyperplane classifiers)

## Entropy
![Alt text](/data/confidence.png?raw=true "Entropy")
