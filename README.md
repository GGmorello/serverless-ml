# serverless-ml
Course ID2223 [Scalable Machine Learning](https://id2223kth.github.io/) @ KTH 

## Contributors
- [Gabriele Morello](https://github.com/GGmorello)
- [Ioannis Theodosiou](https://github.com/GiannisTheo)

## Lab 1

In this project we built a simple serverless application using Hopsworks, Huggingface Spaces, Github Actions and Gradio to predict Iris flower species from a picture and wine quality from a variety of features.

### Iris Flower Species Prediction

Iris UI: https://huggingface.co/spaces/GGmorello/iris

Iris Monitor: https://huggingface.co/spaces/GGmorello/iris-monitor

Code in `lab1/iris/`

### Wine Quality Prediction

Wine UI: https://huggingface.co/spaces/gianTheo/wine

Wine Monitor: https://huggingface.co/spaces/gianTheo/wine-monitor

Code in `lab1/wine/`

Components: 

- `lab1/wine/wine-eda-and-backfill-feature-group.ipynb`: EDA of the wine dataset, feature group creation

- `lab1/wine/wine-training-pipeline.ipynb`: dataset splitting, training using Random Forest, display of results such as accuracy and confusion matrix, export of model to model registry

- `lab1/wine/wine-feature-pipeline-daily.py`: daily script to add a new wine to the feature group

- `lab1/wine/wine-batch-inference-pipeline.py`: runs batch inference on the last wine added to the feature group, creates confusion matrix and shows predicted quality vs actual quality, it adds every result to a monitor feature group

- `lab1/wine/huggingface-spaces-wine/app.py`: Gradio UI for the wine predictor

- `lab1/wine/huggingface-spaces-wine-monitor/app.py`: Gradio UI for the wine monitor

- `.github/workflows/main.yml`: Github Actions workflow to run wine-feature-pipeline-daily.py and wine-batch-inference-pipeline.py every day, (it also includes the Iris workflow)