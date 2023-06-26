---
key: jekyll-text-theme
title: How Do Vision Transformers Work? 논문 정리
excerpt: 'How Do Vision Transformers Work? 논문 읽고 정리하기'
tags: [AI, Computer Vision, Paper, Self-attention, Multi-head self-attention, ViT]
---

# How Do Vision Transformers Work? (ICLR 2022)

## Abstract

- Multi-head self-attentions (MSAs)는 loss landscape를 flatten시킴으로써 accuracy와 generalization을 향상시킴
  - 이러한 향상은 주로 long-range dependency가 아닌 주로 data specificity 때문
  - ViT는 non-convex losses 때문에 고통받음
    - <span style='color:red'>Large datasets와 loss landscape smoothing method가 이런 문제를 완화시킴</span>
- MSA와 Conv는 정반대의 동작을 함
  - MSA는 low-pass filters
  - Conv는 high-pass filters

- Multi-stage neural network에서 stage의 마지막 부분에 MSA를 위치시키면 prediction에서 중요한 역할을 한다.
- 저자들은 위의 관찰에 insight를 얻어 Conv blocks 마지막 부분을 MSA block으로 대체한 AlterNet 제안
  - AlterNet은 large dataset과 small dataset 모두 우월한 성능을 보여주었음 


---

## 1. Introduction

- MSA의 성공에 주요 요인으로 널리 알려진 내용?
  - Weak inductive bias와 long-range dependencies의 포착
  - <span style='color:blue'>Weak inductive bias?</span>
    - <span style='color:blue'>Inductive bias: 학습 시 만나보지 않았던 상황에 대해서 정확한 예측을 하기 위해 사용되는 추가적인 가정</span>
      - <span style='color:blue'>Inductive bias를 가지는 모델 성능 ↑, CNN의 경우 locality (근접 pixel끼리의 종속성), transitional invariance (사물 위치가 바뀌어도 동일 사물로 인식) 등의 특성 때문에 이미지 데이터에 적합한 모델이 됨</span>
      - <span style='color:blue'>반면 CNN이 아닌 fully connected의 경우, 입력 값들이 개별 unit으로 정의되고 이 값들이 모두 연결되어 있는 것으로 가정 → 모든 입력 값이 모든 출력 값에 영향을 미치기 때문에 특별한 relation inductive bias 가정 X, 때문에 weak inductive bias를 가짐</span>
      - <span style='color:blue'>하지만 inductive bias가 강하다고 무조건 학습을 잘하는 것은 아님</span>
        - <span style='color:blue'>왜? 데이터에 대한 가정을 하고 있기 때문이다. 이런 이유 때문에 Vision Transformer보다 낮은 성능을 보임</span>
- 근데 MSA의 over-flexibiity 때문에, MSA로 구성된 ViT는 training dataset에 overfitting 되는 경향이 있고, small dataset에서 낮은 예측 성능이 나온다. (예, CIFAR에서의 이미지 분류)

---

### 1.1 Related Work

작성 중

<br/>

> **Reference** 
>
> 
