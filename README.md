# C3D
Implementation of https://arxiv.org/abs/1412.0767 using pytorch and keras.


## Keras C3D


- The **test.py** code contains output on one video
- **train_ucf101_HDF5.py** contains the code to re-train the network on newdata. Uses HDF5Matrix as dataloaders (Not working)
- **train_ucf_complete_finetune.py** contains the code to re-train the complete network freezing the first 5 layers. It uses manual train_on_batch function to train the network.
- **train_ucf_fc_finetune.py** contains the code to fintune all the fc layers freezing remaining layers.
- **train_ucf_only_final_layer.py** contains the code to fintune the bottleneck layer.
- **utils.py** contains different functions for pre-processing and arranging the data
- **c_models.py** contains network architecture
- **test_untrimmed.py** contains code for predicting the outputs on an untrimmed video.

### List of resources
- Tensorflow Pre-trained Models - https://github.com/hx173149/C3D-tensorflow


### TO RUN
- Download the repo. https://github.com/axon-research/c3d-keras and run bash **do_everything.sh**.
- Download a youtube video
```
pip install youtube-dll
```
- RUN test.py to check if the model is working or not.
- Download the UCF-101 Dataset http://crcv.ucf.edu/data/UCF101.php
- RUN create_datasets.py to create txt_files.
- RUN utils.py to create .npy files (Need 300GB of space)
- RUN train_ucf_only_final_layer.py or anyother by properly specifying the pre-trained model location . (Need a GPU. 1sec per iteration)

Training progress
- We have trained 3 models till now
  - **train_ucf_complete_finetune.py** this is trained on UCF-101 with the first 5 layers freezed. (stopped After 14 epochs)
    - epoch-4  Validation Accuracy: 0.8284964748319397, Validation Loss:0.6974218473589006
    - epoch-5  Validation Accuracy: 0.8617806197737334, Validation Loss:0.5594929565655709
    - epoch-6  Validation Accuracy: 0.8837514346614199, Validation Loss:0.4552247332141234
    - epoch-7  Validation Accuracy: 0.9042465978029185, Validation Loss:0.3830519254048826
    - epoch-8  Validation Accuracy: 0.9167076569929496, Validation Loss:0.32857723246608633
    - epoch-9  Validation Accuracy: 0.9227742252828333, Validation Loss:0.29302591036610437
    - epoch-10 Validation Accuracy: 0.9368748975241843, Validation Loss:0.25210144501089266
    - epoch-11 Validation Accuracy: 0.9429414658140679, Validation Loss:0.22403812064974402
    - epoch-12 Validation Accuracy: 0.9458927693064437, Validation Loss:0.20081407733568646
    - epoch-13 Validation Accuracy: 0.9511395310706673, Validation Loss:0.18801887244554555
    - epoch-14 Validation Accuracy: 0.955894408919495, Validation Loss:0.17157246496051096
  - **train_ucf_fc_finetune.py** this is trained on UCF-101 with fine-tuning only FC layers - (Stopped After 5 epochs)
    - epoch-4 Validation Accuracy: 0.7396704377766847, Validation Loss:1.4196641644485493
    - epoch-5 Validation Accuracy: 0.7726266601082145, Validation Loss:1.1777194898506052
  - **train_ucf_only_final_layer.py** only the final layer is trained with all remaining layers freezed - (Trained still the accuracy became stagnant)
    - epoch-5 Validation Accuracy on external data is 78.7%

- Didn't find **PyTorch** Pre-trained models and find it difficult to train from scratch due to the scarcity of resources. Code can be checked.

**mail to vanapaliprakash@gmail.com for pretrained models or any bug reporting**

PS: Documentation and Code comments are incomplete and verbose. Will refactor them in free-time. Use only if you find it useful.

Thank you.
