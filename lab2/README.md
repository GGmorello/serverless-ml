# Finetune-Whisper Lab2

Course ID2223 [Scalable Machine Learning](https://id2223kth.github.io/) @ KTH 

## Contributors: 

- [Gabriele Morello](https://github.com/GGmorello)
- [Ioannis Theodosiou](https://github.com/GiannisTheo)

## About

This project is an attempt of finetuning Whisper Model from open AI for Automatic Speech Recognition (ASR) in Italian. [Base Model](https://huggingface.co/openai/whisper-small) was obtained from the Hugging Face Hub and the dataset used for finetuning was the [common voice italian subset dataset](https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/it). The whisper model family (available in different sizes) are pretrained models for ASR trained on a dataset of 680k hours of audio labeled with their corresponding transciption. By default Whisper works reasonably well in multilingual speech recognition tasks alongside with speech translation and language identification and does not require thorough finetuning.

## Implementation

Our finetuning method consists of:

- Feature Pipeline: In our feature pipeline we utilize only a small 5% subset of the Common voice Italian dataset. Training was performed on Google Colab platform with a Tesla K80 GPU for a non premium account and we wanted to speed up the process. We utilize both the training and the test splits to perform inference and evaluate performance. We wrap a feature extractor object responsible for padding/truncating the audio to 30s and apply a mel-spectrogram transform and a pretrained Whisper Tokenizer object responsible for post processing our output to text format to a whisperprocessor object. This object doesnt change during training and we can store and reuse it when its needed. The processor object is applied to our dataset which is saved in google drive so the processing is not required again when we resume the training. 

- Training Pipeline: In our training pipeline we load the dataset as a subset of the Italian Common Voice dataset. We also load the small Whisper model from the Whisper family. The training is carried through the trainer method of hugging face and the evaluation metric used is the WER (word error) metric as used in most ASR tasks. 

- User Interface: For demo purposes we implemented a gradio app that allows users to input data and perform inference in 3 ways. Audio directly from microphone , audio from a youtube url and audio from a file upload.


## Hyperparameters
For our training process we used the following hyperparameters in our seq2seqtrainer method:
- per_device_batch_train_size = 16. We used the maximum number that actually was allowed with the GPU that we were allocated by Colab
- gradient_accumulation_steps = 1. This is used for compensating for the decrease in batch_size. Each time we halve the batch size we double the gradient accumulation steps
- learning rate = 1e^-5 This is one of the most important hyperparameters. We chose it by performing multiple experiments.
- warmup_steps = 500. Number of steps used for a linear warmup from 0 to learning_rate.
- num_train_epochs = 1
- evaluation_strategy = "steps".  The evaluation strategy to adopt during training. We perform evaluation every eval_steps =  100. 
- save_strategy = "steps". The save strategy to adopt during training. We save the model after save_steps = 100.
- save_total_limit = 2. will limit the total amount of checkpoints. Deletes the older checkpoints in output_dir. When load_best_model_at_end is enabled, the “best” checkpoint according to metric_for_best_model will always be retained in addition to the most recent ones. 


## Possible Improvements

### Data Centric Approach
- It is possible to improve model performance by increasing the number of training audio hours and their corresponding transciptions. Even when using golab GPU the training is slow and thus we had to use only a small subset as 5% of the Italian dataset. Moreover we face a limit of loading the whole dataset on Google Colab when it surpases 16GB. One solution could be to run the training locally on our own GPU or use Google Cloud.

### Model Centric Approach
- To improve performance we could use an other model from the Whisper Family with larger size. Other possible models are [Whisper Large](https://huggingface.co/openai/whisper-large) and [Whisper Large v2](https://huggingface.co/openai/whisper-large-v2). We only used the small model due to computational and memory restrictions. Model size is also closely related to dataset size as described above. 

- num_train_epochs = 1. This is the number of epochs we used to train our model. We only used one pass of our dataset in order to finish the process in a reasonable time.  We expect that the model with only one epoch of training will not be able to generalize optimally. The WER decreased during train but with more training time we could achieve better results. Setting the number of epochs to 2 will increase performance while keeping low the possibility of the model to overfit.  
- learning_rate = 1e^-5. This is the value of the initial learning rate of the AdamW Optimizer. We could do further experiments to finetune the value towards the default value around 5e^-5 which is the default. The finetuning should be done keeping in mind that high learning rate leads to faster training but increases the risk of overshooting and oscillating around the minimum. Smaller learning rate as the one we chose will help the network converge towards the global minimum but by performing smaller steps and thus increasing the required time.


### Demo on HuggingFace Spaces
[Whisper-it](https://huggingface.co/spaces/gianTheo/Whisper-IT-small)

### Models on HuggingFace

[Whisper-small-it](https://huggingface.co/GGmorello/whisper-small-it)
