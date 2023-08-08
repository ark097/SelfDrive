# SDrive
SDrive is a AI/ML model for an autonomous vehicle in GTA

Lane Detection

1. Run the file LaneDet.py

--------------------------------------------

Object detection

1. Install TensorFlow Object Detection API
2. Run OD_LDv2.py to run lane detection and object detection together

--------------------------------------------

CNN

1. Run collect_data.py with the game running in the background to collect training data
2. Run balance.py to make new set of balanced datasets
2. Run train_model.py to train the inception_v3 model
3. Run test_model.py to test out your model

--------------------------------------------

Important points

1. Collect 1L+ balanced datasets
2. Models.py contains many different CNNs all that can be used for this appliication