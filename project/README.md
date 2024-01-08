# How to run 

1. Run on Colab on CPU the `dogs_cpu.ipynb` notebook to extract images and preprocess pictures.
2. Run on Colab on GPU the `dogs_gpu.ipynb` notebook to train the model and get checkpoints.
3. Run on Colab on CPU the `dogs_inference_and_eval.ipynb` notebook to evaluate the model and see some results.

We did an Huggingface Space to generate images on demand accessible [here](https://huggingface.co/spaces/gianTheo/dog_gans).

To see the full pipeline check `gan_dogs.ipynb`, which is a combination of the three notebooks above and has training logging with losses per epoch and images for each epoch.