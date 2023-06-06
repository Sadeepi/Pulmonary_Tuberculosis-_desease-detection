# Deep learning-based automatic detection of pulmonary tuberculosis disease using chest X-ray images.
##
Tuberculosis is an infectious disease usually caused by Mycobacterium Tuberculosis bacteria. If tuberculosis can be detected at an early stage, patients can get proper treatments and save their 
lives. It is very important to have an effective way to identify Tuberculosis disease in early stage. Medical science has improved very fast in recent years with the Artificial Intelligence and Deep Learning.

#

The datasets used in this study were downloaded from the kaggle.com web site. Two datasets were merged to implement this system. </br>

• Tuberculosis (TB) Chest X-ray Database </br>
• Tuberculosis Chest X-rays (Shenzhen) Dataset </br>

After merging, the newly created dataset includes 3823 Normal images and 734 Tuberculosis images. The dataset is unbalanced.

![9](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/6c9ec39a-4ce9-4306-83cc-75e65f4d55d7)
#
Following are the sample images from the dataset.

![90](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/e9eaa484-636e-4f59-857b-3a471b3b9525)

#
The python libraries like numpy, pandas, OpenCV, matplotlib, seaborn, tensorflow and keras  were used.  Tensorflow is 
a free and open-source software library used to run machine learning, deep learning, and other 
statistical and predictive analytics workloads. Keras is also a high-level neural network library that 
runs on top of Tensorflow.

![importlibraries1](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/81510a4f-5f52-444f-aa8c-3c2824da4337)


#
Jupyter and Google Colab are two environments used to run the python code. Jupyter runs on the 
local machine and uses the RAM and the CPU of the computer while Google Colab runs on google 
server and gives the access to free GPU and TPU. Both Jupyter and Google Colab are used for this 
study. 

#

In this study I used VGG-16 model and sequential model. 


80% of the dataset is used as training data and the remaining 20% is used as test data. Following 
code used to split the data randomly.

![split dataset](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/6c81f69c-f8fb-4b38-adbf-83f46c9391e8)
# 

Creating Sequential Model

Following is the code used to create a sequential model. In this code Dropout layers are used to reduce the overfitting

![Output-sequential_model](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/e43da189-5057-438b-9945-b4365b55f2cc)

VGG - 16 model
It is difficult to satisfy with these results because accuracy and recall values are less than 60. 

VGG - 16 model Results

![29](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/f24a287d-3319-40b1-b959-d158537792fa)

![30](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/3bf0da70-fae4-43ab-9cf6-6f0dab6cacbb)

![31](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/7e5d8afe-6f2d-4570-8c8b-f26dcb7a11aa)

![32](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/7f70a25f-1f58-45a5-84ec-ee02276edb39)



##Sequential model - Undersampling Technique

Undersampling technique is used to balance the dataset. For that normal images were divided into 5 parts and each part merged with tuberculosis images. The model was trained separately for the 
five data sets until successful results were obtained.

![Image- under sampling](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/4bcda0f9-a487-424b-aea9-dd29beb20ba5)

After dividing normal images into five parts, each part contains 734 normal images. Then normal images in each dataset were combined with Tuberculosis images. So finally, each dataset contains 
1468 images.

Following flowchart shows the steps for training the 5 models when using undersampling technique in sequential model.

![flowchart1](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/88fd7e71-f33b-4be9-8386-ac0ce890af51)

Sequential model - Undersampling Technique - Results

First Dataset

![1](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/a79294a3-b72a-485d-bc27-f270e731f804)

#

Second Dataset

![2](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/bbe96cd4-5dab-4c6b-938c-c989375e85bd)
#


Third Dataset

![3](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/4602bf5b-e4d6-4245-86e4-6ac09e9618a3)

#

Fourth Dataset

![4](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/dce078b9-2752-4c72-85a4-1ddc363b2016)

#
Fifth Dataset

![5](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/ee326ba7-203b-4b4a-87c3-e83e332d54bd)

![7](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/f5f173aa-ee5e-493b-921d-b3a08bd96093)

![8](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/74ce0d0a-80da-4f50-8226-ac823cb0f0ab)

![9](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/8a1dc298-5b5c-4336-b71d-475fd46c77cf)

![10](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/ed67cd32-e11c-43cd-87ba-2e3d4620d68b)

## Sequential model - Oversampling Technique

Original dataset having RGB images was used to train the model

![11](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/64750e38-94e7-428c-88a2-bd75aeb6cd87)

The following code is used to oversampling the dataset

![code_balance_data](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/351d4610-31f0-4bbf-9f2e-1b51120baef6)


### 1) Sequential model - Oversampling Technique  using Original Images

The model was trained using original images in the dataset until the successful results were obtained.


![12](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/0352cc43-5eb0-479f-b9b0-b0cdecea61be)


![13](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/afbc2a83-6e46-4411-ba51-8eb53f95497c)


![14](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/b6ba4477-80a3-4bec-9ffc-d815a555e5e4)


![15](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/8b38b4fa-cb1a-423e-8a22-ce91ab69d3f1)


![16](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/8d776db7-c402-4958-afc7-4f3aa2bdd101)


### 2) Sequential model using Grayscale Filter (Oversampling Technique)


![17](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/e79d6e33-c51f-4855-ae54-b33ad1755b02)



![18](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/347ac9de-14c5-49fb-a251-4e68712eed74)



![19](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/c64d3f40-8f8e-4bd4-a53e-9e9e12aa6f15)


![20](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/a9668827-6d52-49f4-bfaf-4b53e21bf955)


![21](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/64dddcd4-94de-43bf-b27e-aa10f51eb0ac)


![22](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/15f2fa20-fc65-4cae-8125-5ce59fac632b)


![23](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/d0b28608-b289-4f47-bcaa-67b5293a6c7b)


### 3) Sequential model using Gaussian Blur Filter (Oversampling Technique)

![24](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/45b218d6-ef61-4360-9932-bd9a5c1275fc)

![25](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/9f1726eb-b3ee-4542-b89d-f1a545f910ea)

![26](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/798d7536-69fa-4e60-9f1e-46292c5c91b5)

![27](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/ac281a4d-d1e9-4871-bbfe-222a86a25c08)

![28](https://github.com/Sadeepi/Pulmonary_Tuberculosis-_desease-detection/assets/86165230/0a717abe-879b-4cb9-88bb-4817ea8b3a0b)




















