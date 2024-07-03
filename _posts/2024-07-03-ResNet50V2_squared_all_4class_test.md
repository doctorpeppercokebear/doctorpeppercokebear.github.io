---
layout: single
title:  "Resnet50v2_squared_all_4class_test"
categories: image classification
tag: [python, blog, Resnetv2, image classification]
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


### 그래프 한글 안깨지는 코드



```python
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt

font_files = font_manager.findSystemFonts(fontpaths='/content/drive/MyDrive/Pal-ette/D2Coding')
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)

plt.rc('font', family='D2Coding')
```

### 라이브러리 불러오기



```python
import tensorflow as tf
import os
from scipy.special import softmax
from PIL import Image
import numpy as np
```

### Data load



```python
!unzip -q "/content/drive/MyDrive/개_안구_tr_te_va_squared_all_4class.zip"
```


```python
test_data = '/content/개_안구_tr_te_va_squared_all_4class/test'
```


```python
model_path = '/content/drive/MyDrive/CV_project/안구질환_모델/Resnet_squared_all_4class_b64_e40.tf'
model = tf.keras.models.load_model(model_path)
```

### 훈련 모델 라벨 지정



```python
labels= [
    '궤양성각막질환',
    '백내장',
    '안검종양',
    '핵경화'
]
```

### 모델 예측



```python
def inference(file_path):
    # 이미지 파일 열기
    img = Image.open(file_path)

    # 이미지를 224x224 픽셀로 크기 조정
    img = img.resize((224, 224))

    # 이미지를 NumPy 배열로 변환
    img = np.array(img)

    # 픽셀 값을 [0, 1] 범위로 정규화
    img = img / 255.

    # 모델 입력 형식에 맞게 이미지의 차원 확장
    img = tf.expand_dims(img, axis=0)

    # 모델을 사용하여 이미지 예측 수행
    pred = model.predict(img, verbose=0)

    # 예측 결과 반환
    return pred
```


```python
# 예측 결과를 저장할 딕셔너리 초기화
predictions = {'target':[], 'pred':[], 'prob':[]}

# 테스트 데이터 디렉토리 내의 각 폴더에 대해 반복
for folder in os.listdir(test_data):
    print(folder)

    # 각 폴더 내의 파일에 대해 반복
    for file in os.listdir(os.path.join(test_data, folder)):
        # 파일 경로를 사용하여 예측 수행
        pred = inference(os.path.join(test_data, folder, file))

        # 실제 라벨을 딕셔너리에 추가
        predictions['target'].append(folder)

        # 예측된 라벨을 딕셔너리에 추가
        predictions['pred'].append(labels[pred.argmax()])

        # 예측 확률을 딕셔너리에 추가
        predictions['prob'].append(pred.tolist())

        # 예측한 샘플 수가 1000의 배수일 때 진행 상황 출력
        if len(predictions['target']) % 1000 == 0:
            print(len(predictions['target']), "predictions done")
```

<pre>
안검종양
궤양성각막질환
1000 predictions done
2000 predictions done
백내장
3000 predictions done
4000 predictions done
핵경화
5000 predictions done
</pre>
### 이미지 개수 확인



```python
import os

base_dir = '/content/개_안구_tr_te_va_squared_all_4class'

for root, dirs, files in os.walk(base_dir):
    if root != base_dir:  # base_dir 자체는 제외
        num_images = sum(1 for file in files if file.endswith('.jpg') or file.endswith('.png'))  # jpg, png 이미지 확인
        print(f"{os.path.basename(root)}: {num_images} images")
```

<pre>
val: 0 images
안검종양: 538 images
궤양성각막질환: 1545 images
백내장: 2321 images
핵경화: 1079 images
test: 0 images
안검종양: 540 images
궤양성각막질환: 1547 images
백내장: 2322 images
핵경화: 1081 images
train: 0 images
안검종양: 4307 images
궤양성각막질환: 12371 images
백내장: 18569 images
핵경화: 8638 images
</pre>

```python
from collections import Counter

Counter(predictions['target'])
```

<pre>
Counter({'안검종양': 540, '궤양성각막질환': 1548, '백내장': 2323, '핵경화': 1081})
</pre>

```python
Counter(predictions['pred'])
```

<pre>
Counter({'안검종양': 661, '핵경화': 998, '백내장': 2244, '궤양성각막질환': 1589})
</pre>

```python
import unicodedata
target = [unicodedata.normalize('NFC', t) for t in predictions['target']]
predictions['target'] = target
```


```python
pred = [unicodedata.normalize('NFC', p) for p in predictions['pred']]
predictions['pred'] = pred
```


```python
print(target[0], len(target[0]))
```

<pre>
안검종양 4
</pre>

```python
!rm -rf ~/.cache/matplotlib
```

### 시각화



```python
# 혼동 행렬 생성 및 시각화
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# 혼동 행렬 생성 및 시각화
# 실제값과 예측값을 사용하여 혼동 행렬 생성 (정규화된 값 사용)
cm = confusion_matrix(predictions['target'], predictions['pred'], normalize='true')

# Seaborn을 사용하여 혼동 행렬을 히트맵으로 시각화
hm = sns.heatmap(cm, annot=True, fmt='.2f', cmap='flare_r')

# x축 레이블 설정
hm.set_xlabel('예측값', fontsize=10)

# y축 레이블 설정
hm.set_ylabel('실제값', fontsize=10)

# x축 틱 레이블 설정 (45도 회전, 오른쪽 정렬)
hm.set_xticklabels(labels=labels, fontsize=10, rotation=45, ha='right', rotation_mode='anchor')

# y축 틱 레이블 설정 (회전 없음)
hm.set_yticklabels(labels=labels, fontsize=10, rotation=0)

# 그래프 출력
plt.show()
```

<pre>
<Figure size 640x480 with 2 Axes>
</pre>

```python
from sklearn.metrics import classification_report

# 실제값과 예측값을 사용하여 분류 보고서 생성
cr = classification_report(predictions['target'], predictions['pred'])

# 분류 보고서 출력
print(cr)
```

<pre>
              precision    recall  f1-score   support

     궤양성각막질환       0.92      0.95      0.94      1548
         백내장       0.84      0.81      0.83      2323
        안검종양       0.70      0.86      0.77       540
         핵경화       0.65      0.60      0.63      1081

    accuracy                           0.81      5492
   macro avg       0.78      0.81      0.79      5492
weighted avg       0.81      0.81      0.81      5492

</pre>