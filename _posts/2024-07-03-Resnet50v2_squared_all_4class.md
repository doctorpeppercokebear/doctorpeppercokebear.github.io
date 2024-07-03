---
layout: single
title:  "Resnet50v2_squared_all_4class"
categories: jupyter
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


# precision > 0.7 이하인 클래스 삭제하고 분류 모델 제작

결막염, 비궤양성각막질환, 색소침착성각막염, 안검내반증, 안검염, 유루증


### 그래프 한글 안 깨지게 하는 코드



```python
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

font_files = font_manager.findSystemFonts(fontpaths='/content/drive/MyDrive/Pal-ette/D2Coding')
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rc('font', family='D2Coding')
```

### 모듈 불러오기




```python
!pip install split-folders
```


```python
import tensorflow as tf
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, AveragePooling2D, concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping # 모델 저장, 조기종료

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

import splitfolders
```

### 데이터 압축풀기



```python
# !unzip -q "/content/drive/MyDrive/CV_project/data/안구질환/개_안구_squared_all.zip"
```


```python
# import os

# base_dir = '/content/개_안구_squared_all'

# for root, dirs, files in os.walk(base_dir):
#     if root != base_dir:  # base_dir 자체는 제외
#         num_images = sum(1 for file in files if file.endswith('.jpg') or file.endswith('.png'))  # jpg, png 이미지 확인
#         print(f"{os.path.basename(root)}: {num_images} images")
```

<pre>
궤양성각막질환: 15463 images
정상: 85139 images
안검염: 7731 images
안검내반증: 10789 images
결막염: 10799 images
비궤양성각막질환: 10797 images
안검종양: 5385 images
핵경화: 10798 images
색소침착성각막염: 7919 images
백내장: 23212 images
유루증: 10796 images
</pre>

```python
# import os
# import shutil

# # 삭제할 파일이 있는 디렉토리 경로
# base_path = "/content/개_안구_squared_all"

# # 삭제할 파일들의 이름 리스트
# files_to_delete = [
#     "결막염",
#     "비궤양성각막질환",
#     "색소침착성각막염",
#     "안검내반증",
#     "안검염",
#     "유루증",
#     "정상"
# ]

# # 파일 삭제
# for file_name in files_to_delete:
#     file_path = os.path.join(base_path, file_name)
#     if os.path.exists(file_path):
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#             print(f"{file_name} 파일을 삭제했습니다.")
#         elif os.path.isdir(file_path):
#             shutil.rmtree(file_path)
#             print(f"{file_name} 디렉토리를 삭제했습니다.")
#     else:
#         print(f"{file_name} 파일이 존재하지 않습니다.")
```

<pre>
결막염 파일이 존재하지 않습니다.
비궤양성각막질환 파일이 존재하지 않습니다.
색소침착성각막염 파일이 존재하지 않습니다.
안검내반증 파일이 존재하지 않습니다.
안검염 파일이 존재하지 않습니다.
유루증 파일이 존재하지 않습니다.
정상 디렉토리를 삭제했습니다.
</pre>

```python
# splitfolders.ratio('/content/개_안구_squared_all', output="개_안구_tr_te_va_squared_all_4class", ratio=(0.8, 0.1, 0.1))  # train/val/test = 8:1:1
```

<pre>
Copying files: 54875 files [00:07, 7464.25 files/s]
</pre>

```python
# prompt: /content/개_안구_tr_te_va_squared_all 이 파일을 압축파일로 드라이브에 저장하는 코드 알려줘

!zip -r /content/drive/MyDrive/개_안구_tr_te_va_squared_all_4class.zip /content/개_안구_tr_te_va_squared_all_4class
```


```python
!unzip -q "/content/drive/MyDrive/개_안구_tr_te_va_squared_all_4class.zip"
```


```python
import os

base_dir = '/content/개_안구_tr_te_va_squared_all_4class'

for root, dirs, files in os.walk(base_dir):
    if root != base_dir:  # base_dir 자체는 제외
        num_images = sum(1 for file in files if file.endswith('.jpg') or file.endswith('.png'))  # jpg, png 이미지 확인
        print(f"{os.path.basename(root)}: {num_images} images")
```

<pre>
train: 0 images
궤양성각막질환: 12371 images
안검종양: 4307 images
핵경화: 8638 images
백내장: 18569 images
test: 0 images
궤양성각막질환: 1547 images
안검종양: 540 images
핵경화: 1081 images
백내장: 2322 images
val: 0 images
궤양성각막질환: 1545 images
안검종양: 538 images
핵경화: 1079 images
백내장: 2321 images
</pre>
### 하이퍼파라미터 설정






```python
batch_size = 64
img_size = 224
learning_rate = 5e-4
epochs = 40


classes_labels= [
    '궤양성각막질환',
    '백내장',
    '안검종양',
    '핵경화',
]

num_classes = len(classes_labels)
base_dir = '/content/개_안구_tr_te_va_squared_all_4class'
```

### 이미지 증강



```python
# 이미지 증강이 적용된 데이터 제너레이터 선언
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,          # 이미지 픽셀 값을 0 ~ 1 사이로 정규화
    width_shift_range=0.1,     # 이미지를 가로로 10% 범위 내에서 무작위 이동
    height_shift_range=0.1,    # 이미지를 세로로 10% 범위 내에서 무작위 이동
    zoom_range=0.1,           # 이미지를 10% 범위 내에서 무작위 확대/축소
    horizontal_flip=True,      # 이미지를 좌우로 무작위 반전
)

validation_datagen = ImageDataGenerator(
    rescale=1. / 255,          # 검증 데이터에도 동일한 정규화 적용
)

test_datagen = ImageDataGenerator(
    rescale=1. / 255          # 테스트 데이터에도 동일한 정규화 적용
)

# flow_from_directory: 디렉토리에서 이미지를 불러와 배치 단위로 제공
train_generator = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'train'),  # 훈련 데이터 디렉토리 경로
    target_size=(img_size, img_size), # 이미지 크기 조정 (224x224)
    batch_size=batch_size,           # 배치 크기 설정 (32)
    class_mode='categorical',        # 다중 클래스 분류이므로 'categorical' 설정
)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(base_dir, 'val'),    # 검증 데이터 디렉토리 경로
    target_size=(img_size, img_size), # 이미지 크기 조정 (224x224)
    batch_size=batch_size,           # 배치 크기 설정 (32)
    class_mode='categorical',        # 다중 클래스 분류이므로 'categorical' 설정
    follow_links=True        # 숨김 파일 무시
)

test_generator = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test'),   # 테스트 데이터 디렉토리 경로
    target_size=(img_size, img_size), # 이미지 크기 조정 (224x224)
    batch_size=batch_size,           # 배치 크기 설정 (32)
    class_mode='categorical'         # 다중 클래스 분류이므로 'categorical' 설정
)
```

<pre>
Found 43891 images belonging to 4 classes.
Found 5483 images belonging to 4 classes.
Found 5491 images belonging to 4 classes.
</pre>
### 이미지 확인



```python
def show_images(generator, num_images=11):
    image, labels = next(generator)
    plt.figure(figsize=(20, 20))  # 이미지 개수에 따라 figsize 조절

    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image[i])

        # 클래스 레이블에 맞는 제목 설정
        class_index = np.argmax(labels[i])  # one-hot encoded labels에서 클래스 인덱스 추출

        # Check if class_index is within the range of classes_labels
        if class_index >= len(classes_labels):
            class_index = 0

        class_name = classes_labels[class_index]
        plt.title(class_name)

        plt.axis('off')
    plt.show()

show_images(train_generator, num_images=11)  # 모든 클래스 이미지 확인
show_images(test_generator, num_images=11)
```

<pre>
<Figure size 2000x2000 with 11 Axes>
</pre>
<pre>
<Figure size 2000x2000 with 11 Axes>
</pre>
## 모델 구현

* ResNet50v2 사용




```python
# 모델 불러오기 및 선언
base_model = tf.keras.applications.ResNet50V2(
    input_shape=(img_size, img_size, 3),  # 입력 이미지 크기 (224x224x3)
    include_top=False,                    # 사전 학습된 모델의 마지막 분류 레이어 제외
    weights='imagenet',                   # ImageNet 데이터셋으로 사전 학습된 가중치 사용
    pooling='avg'                         # 특징 맵을 평균 풀링하여 1차원 벡터로 변환
)

inputs = base_model.input                 # 입력 레이어 설정 (base_model의 입력 사용)
x = tf.keras.layers.Dense(128, activation='relu')(base_model.output)  # 128개 노드의 은닉층 추가 (ReLU 활성화 함수 사용)
x = tf.keras.layers.Dropout(0.1)(x)       # 과적합 방지를 위한 드롭아웃 레이어 추가 (10% 드롭아웃)
outputs = tf.keras.layers.Dense(4, activation='softmax')(x)  # 출력 레이어 설정 (클래스 개수만큼 출력 노드, softmax 활성화 함수 사용)

# 모델 정의
model = tf.keras.Model(inputs=inputs, outputs=outputs)  # 입력과 출력을 연결하여 모델 생성

model.summary()
```

<pre>
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5
94668760/94668760 [==============================] - 5s 0us/step
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 224, 224, 3)]        0         []                            
                                                                                                  
 conv1_pad (ZeroPadding2D)   (None, 230, 230, 3)          0         ['input_1[0][0]']             
                                                                                                  
 conv1_conv (Conv2D)         (None, 112, 112, 64)         9472      ['conv1_pad[0][0]']           
                                                                                                  
 pool1_pad (ZeroPadding2D)   (None, 114, 114, 64)         0         ['conv1_conv[0][0]']          
                                                                                                  
 pool1_pool (MaxPooling2D)   (None, 56, 56, 64)           0         ['pool1_pad[0][0]']           
                                                                                                  
 conv2_block1_preact_bn (Ba  (None, 56, 56, 64)           256       ['pool1_pool[0][0]']          
 tchNormalization)                                                                                
                                                                                                  
 conv2_block1_preact_relu (  (None, 56, 56, 64)           0         ['conv2_block1_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv2_block1_1_conv (Conv2  (None, 56, 56, 64)           4096      ['conv2_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv2_block1_1_bn (BatchNo  (None, 56, 56, 64)           256       ['conv2_block1_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block1_1_relu (Activ  (None, 56, 56, 64)           0         ['conv2_block1_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv2_block1_2_pad (ZeroPa  (None, 58, 58, 64)           0         ['conv2_block1_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv2_block1_2_conv (Conv2  (None, 56, 56, 64)           36864     ['conv2_block1_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv2_block1_2_bn (BatchNo  (None, 56, 56, 64)           256       ['conv2_block1_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block1_2_relu (Activ  (None, 56, 56, 64)           0         ['conv2_block1_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv2_block1_0_conv (Conv2  (None, 56, 56, 256)          16640     ['conv2_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv2_block1_3_conv (Conv2  (None, 56, 56, 256)          16640     ['conv2_block1_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv2_block1_out (Add)      (None, 56, 56, 256)          0         ['conv2_block1_0_conv[0][0]', 
                                                                     'conv2_block1_3_conv[0][0]'] 
                                                                                                  
 conv2_block2_preact_bn (Ba  (None, 56, 56, 256)          1024      ['conv2_block1_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv2_block2_preact_relu (  (None, 56, 56, 256)          0         ['conv2_block2_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv2_block2_1_conv (Conv2  (None, 56, 56, 64)           16384     ['conv2_block2_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv2_block2_1_bn (BatchNo  (None, 56, 56, 64)           256       ['conv2_block2_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block2_1_relu (Activ  (None, 56, 56, 64)           0         ['conv2_block2_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv2_block2_2_pad (ZeroPa  (None, 58, 58, 64)           0         ['conv2_block2_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv2_block2_2_conv (Conv2  (None, 56, 56, 64)           36864     ['conv2_block2_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv2_block2_2_bn (BatchNo  (None, 56, 56, 64)           256       ['conv2_block2_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block2_2_relu (Activ  (None, 56, 56, 64)           0         ['conv2_block2_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv2_block2_3_conv (Conv2  (None, 56, 56, 256)          16640     ['conv2_block2_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv2_block2_out (Add)      (None, 56, 56, 256)          0         ['conv2_block1_out[0][0]',    
                                                                     'conv2_block2_3_conv[0][0]'] 
                                                                                                  
 conv2_block3_preact_bn (Ba  (None, 56, 56, 256)          1024      ['conv2_block2_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv2_block3_preact_relu (  (None, 56, 56, 256)          0         ['conv2_block3_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv2_block3_1_conv (Conv2  (None, 56, 56, 64)           16384     ['conv2_block3_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv2_block3_1_bn (BatchNo  (None, 56, 56, 64)           256       ['conv2_block3_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block3_1_relu (Activ  (None, 56, 56, 64)           0         ['conv2_block3_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv2_block3_2_pad (ZeroPa  (None, 58, 58, 64)           0         ['conv2_block3_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv2_block3_2_conv (Conv2  (None, 28, 28, 64)           36864     ['conv2_block3_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv2_block3_2_bn (BatchNo  (None, 28, 28, 64)           256       ['conv2_block3_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv2_block3_2_relu (Activ  (None, 28, 28, 64)           0         ['conv2_block3_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 max_pooling2d (MaxPooling2  (None, 28, 28, 256)          0         ['conv2_block2_out[0][0]']    
 D)                                                                                               
                                                                                                  
 conv2_block3_3_conv (Conv2  (None, 28, 28, 256)          16640     ['conv2_block3_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv2_block3_out (Add)      (None, 28, 28, 256)          0         ['max_pooling2d[0][0]',       
                                                                     'conv2_block3_3_conv[0][0]'] 
                                                                                                  
 conv3_block1_preact_bn (Ba  (None, 28, 28, 256)          1024      ['conv2_block3_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv3_block1_preact_relu (  (None, 28, 28, 256)          0         ['conv3_block1_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv3_block1_1_conv (Conv2  (None, 28, 28, 128)          32768     ['conv3_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv3_block1_1_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block1_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block1_1_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block1_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block1_2_pad (ZeroPa  (None, 30, 30, 128)          0         ['conv3_block1_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv3_block1_2_conv (Conv2  (None, 28, 28, 128)          147456    ['conv3_block1_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv3_block1_2_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block1_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block1_2_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block1_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block1_0_conv (Conv2  (None, 28, 28, 512)          131584    ['conv3_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv3_block1_3_conv (Conv2  (None, 28, 28, 512)          66048     ['conv3_block1_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv3_block1_out (Add)      (None, 28, 28, 512)          0         ['conv3_block1_0_conv[0][0]', 
                                                                     'conv3_block1_3_conv[0][0]'] 
                                                                                                  
 conv3_block2_preact_bn (Ba  (None, 28, 28, 512)          2048      ['conv3_block1_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv3_block2_preact_relu (  (None, 28, 28, 512)          0         ['conv3_block2_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv3_block2_1_conv (Conv2  (None, 28, 28, 128)          65536     ['conv3_block2_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv3_block2_1_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block2_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block2_1_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block2_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block2_2_pad (ZeroPa  (None, 30, 30, 128)          0         ['conv3_block2_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv3_block2_2_conv (Conv2  (None, 28, 28, 128)          147456    ['conv3_block2_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv3_block2_2_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block2_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block2_2_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block2_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block2_3_conv (Conv2  (None, 28, 28, 512)          66048     ['conv3_block2_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv3_block2_out (Add)      (None, 28, 28, 512)          0         ['conv3_block1_out[0][0]',    
                                                                     'conv3_block2_3_conv[0][0]'] 
                                                                                                  
 conv3_block3_preact_bn (Ba  (None, 28, 28, 512)          2048      ['conv3_block2_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv3_block3_preact_relu (  (None, 28, 28, 512)          0         ['conv3_block3_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv3_block3_1_conv (Conv2  (None, 28, 28, 128)          65536     ['conv3_block3_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv3_block3_1_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block3_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block3_1_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block3_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block3_2_pad (ZeroPa  (None, 30, 30, 128)          0         ['conv3_block3_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv3_block3_2_conv (Conv2  (None, 28, 28, 128)          147456    ['conv3_block3_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv3_block3_2_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block3_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block3_2_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block3_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block3_3_conv (Conv2  (None, 28, 28, 512)          66048     ['conv3_block3_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv3_block3_out (Add)      (None, 28, 28, 512)          0         ['conv3_block2_out[0][0]',    
                                                                     'conv3_block3_3_conv[0][0]'] 
                                                                                                  
 conv3_block4_preact_bn (Ba  (None, 28, 28, 512)          2048      ['conv3_block3_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv3_block4_preact_relu (  (None, 28, 28, 512)          0         ['conv3_block4_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv3_block4_1_conv (Conv2  (None, 28, 28, 128)          65536     ['conv3_block4_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv3_block4_1_bn (BatchNo  (None, 28, 28, 128)          512       ['conv3_block4_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block4_1_relu (Activ  (None, 28, 28, 128)          0         ['conv3_block4_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv3_block4_2_pad (ZeroPa  (None, 30, 30, 128)          0         ['conv3_block4_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv3_block4_2_conv (Conv2  (None, 14, 14, 128)          147456    ['conv3_block4_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv3_block4_2_bn (BatchNo  (None, 14, 14, 128)          512       ['conv3_block4_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv3_block4_2_relu (Activ  (None, 14, 14, 128)          0         ['conv3_block4_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 max_pooling2d_1 (MaxPoolin  (None, 14, 14, 512)          0         ['conv3_block3_out[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv3_block4_3_conv (Conv2  (None, 14, 14, 512)          66048     ['conv3_block4_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv3_block4_out (Add)      (None, 14, 14, 512)          0         ['max_pooling2d_1[0][0]',     
                                                                     'conv3_block4_3_conv[0][0]'] 
                                                                                                  
 conv4_block1_preact_bn (Ba  (None, 14, 14, 512)          2048      ['conv3_block4_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block1_preact_relu (  (None, 14, 14, 512)          0         ['conv4_block1_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block1_1_conv (Conv2  (None, 14, 14, 256)          131072    ['conv4_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block1_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block1_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block1_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block1_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block1_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block1_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block1_2_conv (Conv2  (None, 14, 14, 256)          589824    ['conv4_block1_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block1_2_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block1_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block1_2_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block1_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block1_0_conv (Conv2  (None, 14, 14, 1024)         525312    ['conv4_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block1_3_conv (Conv2  (None, 14, 14, 1024)         263168    ['conv4_block1_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block1_out (Add)      (None, 14, 14, 1024)         0         ['conv4_block1_0_conv[0][0]', 
                                                                     'conv4_block1_3_conv[0][0]'] 
                                                                                                  
 conv4_block2_preact_bn (Ba  (None, 14, 14, 1024)         4096      ['conv4_block1_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block2_preact_relu (  (None, 14, 14, 1024)         0         ['conv4_block2_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block2_1_conv (Conv2  (None, 14, 14, 256)          262144    ['conv4_block2_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block2_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block2_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block2_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block2_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block2_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block2_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block2_2_conv (Conv2  (None, 14, 14, 256)          589824    ['conv4_block2_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block2_2_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block2_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block2_2_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block2_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block2_3_conv (Conv2  (None, 14, 14, 1024)         263168    ['conv4_block2_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block2_out (Add)      (None, 14, 14, 1024)         0         ['conv4_block1_out[0][0]',    
                                                                     'conv4_block2_3_conv[0][0]'] 
                                                                                                  
 conv4_block3_preact_bn (Ba  (None, 14, 14, 1024)         4096      ['conv4_block2_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block3_preact_relu (  (None, 14, 14, 1024)         0         ['conv4_block3_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block3_1_conv (Conv2  (None, 14, 14, 256)          262144    ['conv4_block3_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block3_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block3_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block3_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block3_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block3_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block3_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block3_2_conv (Conv2  (None, 14, 14, 256)          589824    ['conv4_block3_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block3_2_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block3_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block3_2_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block3_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block3_3_conv (Conv2  (None, 14, 14, 1024)         263168    ['conv4_block3_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block3_out (Add)      (None, 14, 14, 1024)         0         ['conv4_block2_out[0][0]',    
                                                                     'conv4_block3_3_conv[0][0]'] 
                                                                                                  
 conv4_block4_preact_bn (Ba  (None, 14, 14, 1024)         4096      ['conv4_block3_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block4_preact_relu (  (None, 14, 14, 1024)         0         ['conv4_block4_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block4_1_conv (Conv2  (None, 14, 14, 256)          262144    ['conv4_block4_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block4_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block4_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block4_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block4_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block4_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block4_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block4_2_conv (Conv2  (None, 14, 14, 256)          589824    ['conv4_block4_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block4_2_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block4_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block4_2_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block4_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block4_3_conv (Conv2  (None, 14, 14, 1024)         263168    ['conv4_block4_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block4_out (Add)      (None, 14, 14, 1024)         0         ['conv4_block3_out[0][0]',    
                                                                     'conv4_block4_3_conv[0][0]'] 
                                                                                                  
 conv4_block5_preact_bn (Ba  (None, 14, 14, 1024)         4096      ['conv4_block4_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block5_preact_relu (  (None, 14, 14, 1024)         0         ['conv4_block5_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block5_1_conv (Conv2  (None, 14, 14, 256)          262144    ['conv4_block5_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block5_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block5_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block5_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block5_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block5_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block5_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block5_2_conv (Conv2  (None, 14, 14, 256)          589824    ['conv4_block5_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block5_2_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block5_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block5_2_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block5_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block5_3_conv (Conv2  (None, 14, 14, 1024)         263168    ['conv4_block5_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block5_out (Add)      (None, 14, 14, 1024)         0         ['conv4_block4_out[0][0]',    
                                                                     'conv4_block5_3_conv[0][0]'] 
                                                                                                  
 conv4_block6_preact_bn (Ba  (None, 14, 14, 1024)         4096      ['conv4_block5_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv4_block6_preact_relu (  (None, 14, 14, 1024)         0         ['conv4_block6_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv4_block6_1_conv (Conv2  (None, 14, 14, 256)          262144    ['conv4_block6_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv4_block6_1_bn (BatchNo  (None, 14, 14, 256)          1024      ['conv4_block6_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block6_1_relu (Activ  (None, 14, 14, 256)          0         ['conv4_block6_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv4_block6_2_pad (ZeroPa  (None, 16, 16, 256)          0         ['conv4_block6_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv4_block6_2_conv (Conv2  (None, 7, 7, 256)            589824    ['conv4_block6_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv4_block6_2_bn (BatchNo  (None, 7, 7, 256)            1024      ['conv4_block6_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv4_block6_2_relu (Activ  (None, 7, 7, 256)            0         ['conv4_block6_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 max_pooling2d_2 (MaxPoolin  (None, 7, 7, 1024)           0         ['conv4_block5_out[0][0]']    
 g2D)                                                                                             
                                                                                                  
 conv4_block6_3_conv (Conv2  (None, 7, 7, 1024)           263168    ['conv4_block6_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv4_block6_out (Add)      (None, 7, 7, 1024)           0         ['max_pooling2d_2[0][0]',     
                                                                     'conv4_block6_3_conv[0][0]'] 
                                                                                                  
 conv5_block1_preact_bn (Ba  (None, 7, 7, 1024)           4096      ['conv4_block6_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv5_block1_preact_relu (  (None, 7, 7, 1024)           0         ['conv5_block1_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv5_block1_1_conv (Conv2  (None, 7, 7, 512)            524288    ['conv5_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv5_block1_1_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block1_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block1_1_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block1_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block1_2_pad (ZeroPa  (None, 9, 9, 512)            0         ['conv5_block1_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv5_block1_2_conv (Conv2  (None, 7, 7, 512)            2359296   ['conv5_block1_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv5_block1_2_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block1_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block1_2_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block1_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block1_0_conv (Conv2  (None, 7, 7, 2048)           2099200   ['conv5_block1_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv5_block1_3_conv (Conv2  (None, 7, 7, 2048)           1050624   ['conv5_block1_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv5_block1_out (Add)      (None, 7, 7, 2048)           0         ['conv5_block1_0_conv[0][0]', 
                                                                     'conv5_block1_3_conv[0][0]'] 
                                                                                                  
 conv5_block2_preact_bn (Ba  (None, 7, 7, 2048)           8192      ['conv5_block1_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv5_block2_preact_relu (  (None, 7, 7, 2048)           0         ['conv5_block2_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv5_block2_1_conv (Conv2  (None, 7, 7, 512)            1048576   ['conv5_block2_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv5_block2_1_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block2_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block2_1_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block2_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block2_2_pad (ZeroPa  (None, 9, 9, 512)            0         ['conv5_block2_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv5_block2_2_conv (Conv2  (None, 7, 7, 512)            2359296   ['conv5_block2_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv5_block2_2_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block2_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block2_2_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block2_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block2_3_conv (Conv2  (None, 7, 7, 2048)           1050624   ['conv5_block2_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv5_block2_out (Add)      (None, 7, 7, 2048)           0         ['conv5_block1_out[0][0]',    
                                                                     'conv5_block2_3_conv[0][0]'] 
                                                                                                  
 conv5_block3_preact_bn (Ba  (None, 7, 7, 2048)           8192      ['conv5_block2_out[0][0]']    
 tchNormalization)                                                                                
                                                                                                  
 conv5_block3_preact_relu (  (None, 7, 7, 2048)           0         ['conv5_block3_preact_bn[0][0]
 Activation)                                                        ']                            
                                                                                                  
 conv5_block3_1_conv (Conv2  (None, 7, 7, 512)            1048576   ['conv5_block3_preact_relu[0][
 D)                                                                 0]']                          
                                                                                                  
 conv5_block3_1_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block3_1_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block3_1_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block3_1_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block3_2_pad (ZeroPa  (None, 9, 9, 512)            0         ['conv5_block3_1_relu[0][0]'] 
 dding2D)                                                                                         
                                                                                                  
 conv5_block3_2_conv (Conv2  (None, 7, 7, 512)            2359296   ['conv5_block3_2_pad[0][0]']  
 D)                                                                                               
                                                                                                  
 conv5_block3_2_bn (BatchNo  (None, 7, 7, 512)            2048      ['conv5_block3_2_conv[0][0]'] 
 rmalization)                                                                                     
                                                                                                  
 conv5_block3_2_relu (Activ  (None, 7, 7, 512)            0         ['conv5_block3_2_bn[0][0]']   
 ation)                                                                                           
                                                                                                  
 conv5_block3_3_conv (Conv2  (None, 7, 7, 2048)           1050624   ['conv5_block3_2_relu[0][0]'] 
 D)                                                                                               
                                                                                                  
 conv5_block3_out (Add)      (None, 7, 7, 2048)           0         ['conv5_block2_out[0][0]',    
                                                                     'conv5_block3_3_conv[0][0]'] 
                                                                                                  
 post_bn (BatchNormalizatio  (None, 7, 7, 2048)           8192      ['conv5_block3_out[0][0]']    
 n)                                                                                               
                                                                                                  
 post_relu (Activation)      (None, 7, 7, 2048)           0         ['post_bn[0][0]']             
                                                                                                  
 avg_pool (GlobalAveragePoo  (None, 2048)                 0         ['post_relu[0][0]']           
 ling2D)                                                                                          
                                                                                                  
 dense (Dense)               (None, 128)                  262272    ['avg_pool[0][0]']            
                                                                                                  
 dropout (Dropout)           (None, 128)                  0         ['dense[0][0]']               
                                                                                                  
 dense_1 (Dense)             (None, 4)                    516       ['dropout[0][0]']             
                                                                                                  
==================================================================================================
Total params: 23827588 (90.90 MB)
Trainable params: 23782148 (90.72 MB)
Non-trainable params: 45440 (177.50 KB)
__________________________________________________________________________________________________
</pre>
### 모델 컴파일링




```python
# ophtimizer 설정
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 콜백함수
callback = [ReduceLROnPlateau(monitor='val_loss', mode = 'min', factor=0.1, patience=4, min_lr=1e-7, verbose=1),
            ModelCheckpoint('/content/drive/MyDrive/CV_project/안구질환_모델/Resnet_squared_all_4class_b64_e40.tf', monitor='val_loss', mode='min', save_best_only=True)]
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True, mode="min", monitor='val_loss')  # 과대 적합 방지 및 , 필요하지 않은 훈련 하지 않도록 조기 종료

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)
```

## 모델 학습'




```python
histroy = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[callback, early_stopping_cb]
)
```

<pre>
Epoch 1/40
686/686 [==============================] - 540s 739ms/step - loss: 0.3393 - accuracy: 0.6752 - precision_2: 0.7183 - recall_2: 0.6214 - auc_2: 0.8921 - val_loss: 0.3504 - val_accuracy: 0.6677 - val_precision_2: 0.6876 - val_recall_2: 0.6520 - val_auc_2: 0.8891 - lr: 5.0000e-04
Epoch 2/40
686/686 [==============================] - 495s 721ms/step - loss: 0.3018 - accuracy: 0.7165 - precision_2: 0.7486 - recall_2: 0.6767 - auc_2: 0.9157 - val_loss: 0.3337 - val_accuracy: 0.6830 - val_precision_2: 0.7077 - val_recall_2: 0.6619 - val_auc_2: 0.8977 - lr: 5.0000e-04
Epoch 3/40
686/686 [==============================] - 495s 721ms/step - loss: 0.2830 - accuracy: 0.7345 - precision_2: 0.7625 - recall_2: 0.7034 - auc_2: 0.9264 - val_loss: 0.2888 - val_accuracy: 0.7301 - val_precision_2: 0.7560 - val_recall_2: 0.7071 - val_auc_2: 0.9229 - lr: 5.0000e-04
Epoch 4/40
686/686 [==============================] - 495s 721ms/step - loss: 0.2688 - accuracy: 0.7492 - precision_2: 0.7739 - recall_2: 0.7224 - auc_2: 0.9337 - val_loss: 0.2849 - val_accuracy: 0.7372 - val_precision_2: 0.7611 - val_recall_2: 0.7100 - val_auc_2: 0.9241 - lr: 5.0000e-04
Epoch 5/40
686/686 [==============================] - 497s 724ms/step - loss: 0.2576 - accuracy: 0.7598 - precision_2: 0.7830 - recall_2: 0.7333 - auc_2: 0.9392 - val_loss: 0.2731 - val_accuracy: 0.7441 - val_precision_2: 0.7714 - val_recall_2: 0.7113 - val_auc_2: 0.9314 - lr: 5.0000e-04
Epoch 6/40
686/686 [==============================] - 474s 690ms/step - loss: 0.2474 - accuracy: 0.7703 - precision_2: 0.7900 - recall_2: 0.7493 - auc_2: 0.9440 - val_loss: 0.2859 - val_accuracy: 0.7314 - val_precision_2: 0.7639 - val_recall_2: 0.6887 - val_auc_2: 0.9246 - lr: 5.0000e-04
Epoch 7/40
686/686 [==============================] - 497s 724ms/step - loss: 0.2409 - accuracy: 0.7757 - precision_2: 0.7951 - recall_2: 0.7542 - auc_2: 0.9469 - val_loss: 0.2529 - val_accuracy: 0.7571 - val_precision_2: 0.7759 - val_recall_2: 0.7343 - val_auc_2: 0.9414 - lr: 5.0000e-04
Epoch 8/40
686/686 [==============================] - 482s 702ms/step - loss: 0.2322 - accuracy: 0.7833 - precision_2: 0.7999 - recall_2: 0.7648 - auc_2: 0.9506 - val_loss: 0.3308 - val_accuracy: 0.7113 - val_precision_2: 0.7365 - val_recall_2: 0.6836 - val_auc_2: 0.9044 - lr: 5.0000e-04
Epoch 9/40
686/686 [==============================] - 477s 694ms/step - loss: 0.2252 - accuracy: 0.7896 - precision_2: 0.8058 - recall_2: 0.7722 - auc_2: 0.9535 - val_loss: 0.2590 - val_accuracy: 0.7541 - val_precision_2: 0.7713 - val_recall_2: 0.7357 - val_auc_2: 0.9381 - lr: 5.0000e-04
Epoch 10/40
686/686 [==============================] - 497s 724ms/step - loss: 0.2189 - accuracy: 0.7983 - precision_2: 0.8129 - recall_2: 0.7811 - auc_2: 0.9560 - val_loss: 0.2423 - val_accuracy: 0.7753 - val_precision_2: 0.7945 - val_recall_2: 0.7523 - val_auc_2: 0.9462 - lr: 5.0000e-04
Epoch 11/40
686/686 [==============================] - 486s 708ms/step - loss: 0.2118 - accuracy: 0.8010 - precision_2: 0.8149 - recall_2: 0.7864 - auc_2: 0.9588 - val_loss: 0.2447 - val_accuracy: 0.7706 - val_precision_2: 0.7855 - val_recall_2: 0.7489 - val_auc_2: 0.9443 - lr: 5.0000e-04
Epoch 12/40
686/686 [==============================] - 497s 724ms/step - loss: 0.2064 - accuracy: 0.8068 - precision_2: 0.8208 - recall_2: 0.7920 - auc_2: 0.9609 - val_loss: 0.2357 - val_accuracy: 0.7806 - val_precision_2: 0.7980 - val_recall_2: 0.7658 - val_auc_2: 0.9475 - lr: 5.0000e-04
Epoch 13/40
686/686 [==============================] - 478s 697ms/step - loss: 0.2014 - accuracy: 0.8117 - precision_2: 0.8244 - recall_2: 0.7982 - auc_2: 0.9625 - val_loss: 0.2629 - val_accuracy: 0.7607 - val_precision_2: 0.7732 - val_recall_2: 0.7459 - val_auc_2: 0.9365 - lr: 5.0000e-04
Epoch 14/40
686/686 [==============================] - 473s 689ms/step - loss: 0.1961 - accuracy: 0.8162 - precision_2: 0.8284 - recall_2: 0.8039 - auc_2: 0.9644 - val_loss: 0.2469 - val_accuracy: 0.7693 - val_precision_2: 0.7867 - val_recall_2: 0.7529 - val_auc_2: 0.9437 - lr: 5.0000e-04
Epoch 15/40
686/686 [==============================] - 477s 696ms/step - loss: 0.1910 - accuracy: 0.8201 - precision_2: 0.8316 - recall_2: 0.8075 - auc_2: 0.9662 - val_loss: 0.2470 - val_accuracy: 0.7737 - val_precision_2: 0.7876 - val_recall_2: 0.7604 - val_auc_2: 0.9426 - lr: 5.0000e-04
Epoch 16/40
686/686 [==============================] - ETA: 0s - loss: 0.1872 - accuracy: 0.8228 - precision_2: 0.8343 - recall_2: 0.8111 - auc_2: 0.9674
Epoch 16: ReduceLROnPlateau reducing learning rate to 5.0000002374872565e-05.
686/686 [==============================] - 482s 703ms/step - loss: 0.1872 - accuracy: 0.8228 - precision_2: 0.8343 - recall_2: 0.8111 - auc_2: 0.9674 - val_loss: 0.2468 - val_accuracy: 0.7777 - val_precision_2: 0.7877 - val_recall_2: 0.7655 - val_auc_2: 0.9431 - lr: 5.0000e-04
Epoch 17/40
686/686 [==============================] - 509s 741ms/step - loss: 0.1618 - accuracy: 0.8448 - precision_2: 0.8532 - recall_2: 0.8368 - auc_2: 0.9759 - val_loss: 0.2152 - val_accuracy: 0.8012 - val_precision_2: 0.8076 - val_recall_2: 0.7955 - val_auc_2: 0.9557 - lr: 5.0000e-05
Epoch 18/40
686/686 [==============================] - 507s 738ms/step - loss: 0.1509 - accuracy: 0.8534 - precision_2: 0.8609 - recall_2: 0.8468 - auc_2: 0.9788 - val_loss: 0.2129 - val_accuracy: 0.8028 - val_precision_2: 0.8098 - val_recall_2: 0.7965 - val_auc_2: 0.9558 - lr: 5.0000e-05
Epoch 19/40
686/686 [==============================] - 487s 710ms/step - loss: 0.1482 - accuracy: 0.8542 - precision_2: 0.8610 - recall_2: 0.8473 - auc_2: 0.9794 - val_loss: 0.2190 - val_accuracy: 0.8058 - val_precision_2: 0.8104 - val_recall_2: 0.8007 - val_auc_2: 0.9539 - lr: 5.0000e-05
Epoch 20/40
686/686 [==============================] - 502s 731ms/step - loss: 0.1450 - accuracy: 0.8584 - precision_2: 0.8647 - recall_2: 0.8528 - auc_2: 0.9803 - val_loss: 0.2112 - val_accuracy: 0.8027 - val_precision_2: 0.8089 - val_recall_2: 0.7959 - val_auc_2: 0.9572 - lr: 5.0000e-05
Epoch 21/40
686/686 [==============================] - 482s 702ms/step - loss: 0.1428 - accuracy: 0.8592 - precision_2: 0.8660 - recall_2: 0.8525 - auc_2: 0.9809 - val_loss: 0.2150 - val_accuracy: 0.8070 - val_precision_2: 0.8126 - val_recall_2: 0.8005 - val_auc_2: 0.9554 - lr: 5.0000e-05
Epoch 22/40
686/686 [==============================] - 482s 702ms/step - loss: 0.1413 - accuracy: 0.8605 - precision_2: 0.8668 - recall_2: 0.8538 - auc_2: 0.9813 - val_loss: 0.2167 - val_accuracy: 0.8030 - val_precision_2: 0.8069 - val_recall_2: 0.7979 - val_auc_2: 0.9545 - lr: 5.0000e-05
Epoch 23/40
686/686 [==============================] - 485s 707ms/step - loss: 0.1382 - accuracy: 0.8644 - precision_2: 0.8708 - recall_2: 0.8585 - auc_2: 0.9818 - val_loss: 0.2164 - val_accuracy: 0.7992 - val_precision_2: 0.8057 - val_recall_2: 0.7934 - val_auc_2: 0.9556 - lr: 5.0000e-05
Epoch 24/40
686/686 [==============================] - ETA: 0s - loss: 0.1366 - accuracy: 0.8653 - precision_2: 0.8723 - recall_2: 0.8591 - auc_2: 0.9824
Epoch 24: ReduceLROnPlateau reducing learning rate to 5.000000237487257e-06.
686/686 [==============================] - 479s 698ms/step - loss: 0.1366 - accuracy: 0.8653 - precision_2: 0.8723 - recall_2: 0.8591 - auc_2: 0.9824 - val_loss: 0.2159 - val_accuracy: 0.8059 - val_precision_2: 0.8107 - val_recall_2: 0.8016 - val_auc_2: 0.9551 - lr: 5.0000e-05
Epoch 25/40
686/686 [==============================] - 483s 703ms/step - loss: 0.1312 - accuracy: 0.8694 - precision_2: 0.8757 - recall_2: 0.8637 - auc_2: 0.9840 - val_loss: 0.2188 - val_accuracy: 0.8069 - val_precision_2: 0.8115 - val_recall_2: 0.8032 - val_auc_2: 0.9548 - lr: 5.0000e-06
Epoch 26/40
686/686 [==============================] - 485s 706ms/step - loss: 0.1316 - accuracy: 0.8691 - precision_2: 0.8751 - recall_2: 0.8636 - auc_2: 0.9837 - val_loss: 0.2187 - val_accuracy: 0.8043 - val_precision_2: 0.8098 - val_recall_2: 0.8008 - val_auc_2: 0.9545 - lr: 5.0000e-06
Epoch 27/40
686/686 [==============================] - 485s 707ms/step - loss: 0.1307 - accuracy: 0.8697 - precision_2: 0.8754 - recall_2: 0.8642 - auc_2: 0.9839 - val_loss: 0.2192 - val_accuracy: 0.8043 - val_precision_2: 0.8090 - val_recall_2: 0.8005 - val_auc_2: 0.9545 - lr: 5.0000e-06
Epoch 28/40
686/686 [==============================] - ETA: 0s - loss: 0.1305 - accuracy: 0.8684 - precision_2: 0.8743 - recall_2: 0.8633 - auc_2: 0.9840
Epoch 28: ReduceLROnPlateau reducing learning rate to 5.000000328436726e-07.
686/686 [==============================] - 484s 705ms/step - loss: 0.1305 - accuracy: 0.8684 - precision_2: 0.8743 - recall_2: 0.8633 - auc_2: 0.9840 - val_loss: 0.2202 - val_accuracy: 0.8045 - val_precision_2: 0.8087 - val_recall_2: 0.8003 - val_auc_2: 0.9541 - lr: 5.0000e-06
Epoch 29/40
686/686 [==============================] - 481s 701ms/step - loss: 0.1306 - accuracy: 0.8689 - precision_2: 0.8747 - recall_2: 0.8634 - auc_2: 0.9839 - val_loss: 0.2201 - val_accuracy: 0.8038 - val_precision_2: 0.8091 - val_recall_2: 0.8001 - val_auc_2: 0.9541 - lr: 5.0000e-07
Epoch 30/40
686/686 [==============================] - 477s 695ms/step - loss: 0.1293 - accuracy: 0.8712 - precision_2: 0.8775 - recall_2: 0.8658 - auc_2: 0.9843 - val_loss: 0.2197 - val_accuracy: 0.8045 - val_precision_2: 0.8093 - val_recall_2: 0.8001 - val_auc_2: 0.9544 - lr: 5.0000e-07
</pre>
## 평가지표




```python
# 학습 로그 출력
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(40), histroy.history['loss'], label='Train Loss', color="blue")
plt.plot(range(40), histroy.history['val_loss'], label='Validation Loss', color="red")
plt.title('Traning and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(40), histroy.history['accuracy'], label='Train Accuracy', color="blue")
plt.plot(range(40), histroy.history['val_accuracy'], label='Validation Accuracy', color="red")
plt.title('Traning and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
```


```python
# prompt: .h5 모델로 저장하는 코드 알려줘

model.save('/content/drive/MyDrive/CV_project/안구질환_모델/Resnet_squared_all.h5')
```

<pre>
/usr/local/lib/python3.10/dist-packages/keras/src/engine/training.py:3103: UserWarning: You are saving your model as an HDF5 file via `model.save()`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')`.
  saving_api.save_model(
</pre>

```python
trained_model = tf.keras.models.load_model('/content/drive/MyDrive/CV_project/안구질환_모델/Resnet_squared_all_4class.tf')

trained_model.evaluate(test_generator)
```

<pre>
172/172 [==============================] - 10s 54ms/step - loss: 0.2362 - accuracy: 0.7864 - precision: 0.7992 - recall: 0.7725 - auc: 0.9472
</pre>
<pre>
[0.23616932332515717,
 0.7863777279853821,
 0.7991710901260376,
 0.7725368738174438,
 0.9472035765647888]
</pre>

```python
# prompt: /content/drive/MyDrive/CV_project/안구질환_모델/ 이 경로의 모델을 저장하는 코드 알려줘

model.save('/content/drive/MyDrive/CV_project/안구질환_모델/Resnet_squared_e30_b32_model.tf')
```
