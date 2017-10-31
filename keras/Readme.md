## Keras C3D

- Download the repo. https://github.com/axon-research/c3d-keras and run bash **do_everything.sh**.
- Download a youtube video
```
pip install youtube-dll
```
- The **test.py** code contains output on one video
- **train_ucf101.py** contains the code to re-train the network on newdata. Uses HDF5Matrix as dataloaders (Not working)
- **train_ucf_manual.py** contains the code to re-train the network on newdata. It uses manual train_on_batch function to train the network
- **utils.py** contains different functions for pre-processing and arranging the data
- **c_models.py** contains network architecture


### List of resources
- Tensorflow Pre-trained Models - https://github.com/hx173149/C3D-tensorflow
