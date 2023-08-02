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

- CNN관점에서 보면, MSAs는 large-sized 그리고 data-specific kernel를 가진 모든 feature map points의 transformation이다.
  - MSAs가 CNN처럼 동작한다는 보장은 없지만 convolution layer로 표현은 가능
- Long-range dependencies modeling과 같은 MSA의 weak inductive bias가 예측성능에 도움이 될까??
  - 오히려 적절한 제약 조건을 가진 모델이 강력한 표현을 학습하는데 도움이 됨
    - ex) Small window 내에서만 self-attention를 수행하는 local MSA (window 크기로 제약조건을 줌)는 작은 크기의 dataset뿐만 아니라 크기가 큰 대규모 dataset (ImageNet-21K)에서도 global MSA보다 더 나은 성능을 보여줌, **Swin Transformer를 생각해보면 될 것 같다.**
- 이전 연구에서 MSA가 다음과 같은 흥미로운 속성을 가진다는 사실을 관찰
  1. MSA는 CNN의 예측 성능을 향상시킴 그리고 ViT는 uncertainty를 잘 추정함 (<span style='color:red'>uncertainty를 잘 추정한다는 것은 어떤 의미일까? 일반화 성능이 좋아진다는 것을 의미할까? 추후에 더 공부해보자</span>)
  2. ViT는 data corruptions, image occlusions 그리고 adversarial attacks에 대해서 강인하다. (<span style='color:red'>현재 내가 연구하는 task에서 너무 중요한 관측 사실같다. 해당 연구에 대해 더 알아볼 필요가 있다.</span>)
  3. 마지막 layer에 가까운 MSA는 예측 성능을 크게 향상시킨다.
- 위 3가지 관측 사실에 기반해 저자들은 다음과 같은 의문들을 가지게 된다.
  1. MSA의 특성이 Neural Networks을 최적화가 더 잘되기위해 필요할까? <br />MSA의 long-range dependencies가 Neural Networks 학습에 도움이 될까?
  2. MSA가 Conv처럼 동작할까? <br />아니라면 MSA와 Conv는 얼마나 다를까?
  3. 어떻게 MSA와 Conv를 조합할 수 있을까?<br />MSA와 Conv의 장점을 활용할 수 있을까?
- $z_j=\sum_i{Softmax(\frac{QK}{\sqrt{d}})_i}V_{i,j}$,        (1)
  - Self-attention은 normalized importance와 함께 (spatial) token들을 집계한다.
  - 식 (1)에서도 알 수 있듯이 MSA는 positive importance-weights로 feature map값을 평균화하기 때문에 MSA를 feature map의 학습 가능한 spatial smoothing이라 표현하여 작동원리를 설명할 수 있다.
    - $2\times2$ blur와 같은 학습할 수 없는 spatial smoothing → CNN이 feature를 더 잘 볼 수 있도록 도와줌 (Park & Kim, 2022)
  - 이런 spatial smoothing은 feature map point를 공간적으로 조합하고 loss landscape를 smoothing하여 accuracy뿐만 아니라 robust도 향상시킴
  - Spatial smoothing은 앞서 설명한 MSA (1~3)의 속성을 가지고 있음
- <span style='color:red'>**"MSAs behave like spatial smoothing"**</span>
  - Subsampling layer 전 spatial smoothing은 CNN이 더 잘 볼 수 있게 해준다!
    - Park & Kim (2022) "Blurs behave like ensembles: Spatial smoothings to improve accuracy, uncertainty, and robustness."
      - 해당 논문에서 feature map points의 spatial ensemble때문에 성능 향상이 됬다는 것을 보여줌
      - "Bayesian ensemble average or predictions for proximate data point"을 사용한다.<br/>이는 feature map의 distribution과 같은 data uncertainty뿐만 아니라 Neural Network weights의 posterior distribution과 같은 model uncertainty를 사용한다. <br/>
        $p(z_j|x_j, \mathcal{D}) \simeq \sum_i{\pi(x_i|x_j)p(z_j|x_i, w_i)}$,     (2) <br />
        (<span style='color:red'>정확히 알아볼 필요있음</span>)
        - $\pi(x_i|x_j)$ → 또다른 feature map point $x_j$에 대한 feature map point $x_i$의 normalized importance weight.<br/>즉, $\sum_i{\pi(x_i|x_j)}=1$<br/>Importance → $x_i$와 $x_j$ 사이의 similarity<br/>$p(z_j|x_i, w_i)$ → NN prediction, $p(z_j|x_j, \mathcal{D})$ → output predictive distribution<br/>$w_i$ → training dataset $\mathcal{D}$에 대한 posterior $p(w|\mathcal{D})$에서부터의 NN weight sample
    - 식 (2)는 data point간의 유사성을 기반으로 prediction을 다른 predicrion으로 공간적으로 보완해줌
      - ex) $2\times2$ box blur는 각각 중요도가 1/4인 4개의 인접한 feature map point를 공간적으로 조합!
  - 저자들
    - Self-attention 공식과 인접한 data points에 대한 ensemble averaging 공식이 서로 동일하다는 점에 주목!
    - 식 (1)에서 softmax term과 $V$는 각각 식 (2)에서 다음과 같은 수식과 동일
      - Softmax term = $\pi(x_i|x_j)$
      - $V=p(z_j|x_i, w_i)$
    - 식 (2)에서 weight samples는 MSA의 Multi-heads에 해당
  - Spatial smoothing의 속성은 MSA의 속성과 동일
    1. Spatial smoothing은 CNN의 accuracy를 향상시킴 + spatial smoothing은 잘 계산된 uncertainty를 예측함
    2. Spatial smoothing은 MC dropout (image occulsion과 동일), data corruption, 그리고 adversarial attacks에 대해 강인하고 특히 high-frequency noise에 대해서 강인하다.
    3. Spatial smoothing layer는 output layer와 가까울수록 예측 성능이 크게 향상된다.
  - Spatial Smoothing은 다음과 같은 방법들로 성능을 향상시킨다.
    1. Spatial smoothing은 loss landscape를 평활화함으로써 NN optimization에서 도움이 된다.
    2. Spatial smoothing은 low-pass filter이다.<br/>CNNs는 high-frequency noise에 취약하지만 spatial smoothing은 이러한 noise를 크게 줄임으로써 high-frequency noise에 대한 robust가 향상된다.
    3. Spatial smoothing은 stage의 마지막에 적용했을때 효과적이다.<br/>**왜?** spatial smoothing이 모든 transformed feature map들을 합쳐주기 때문이다.
  - "Metaformer is actullay what you need for vision"에서 ViT의 MSA layer를 non-trainable average pooling layer로 대체 가능하다고 설명

### 1.2 Contribution

- **① What properties of MSAs do we need to improve optimization?**

  - MSA = 일반화된 spatial smoothing

    - 어떤 의미? MSA의 식 (1)이 적절한 inductive bias이기 때문에 성능이 향상됨!

  - Weak inductive bias는 NN학습에 방해됨

  - MSA의 핵심 특징은 long-range dependency가 아닌 데이터 특성이다.

    - 극단적인 예시) $3\times3$ receptive field를 가진 local MSA는 불필요한 자유도를 줄이기 때문에 global MSA보다 성능이 뛰어남

  - MSA의 장점과 단점

    ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/0b1880d1-f1fc-462a-9c43-ab6ad8a46f07)

    - 장점?
      - loss landscape가 평탄화될수록 더 나은 성능과 일반화 성능을 가짐
        - MSA는 크기가 큰 data에서 accuracy 뿐만 아니라 robustness도 향상된다!
    - 단점?
      - MSA는 작은 data에서 negative hessian eigenvalue을 허용
        - 무슨 말??? MSA의 loss landscape가 non-convexity하다는 말
          - 이런 non-convexity는 NN최적화를 방해!
        - 대량의 학습 데이터는 negative eigenvalues를 억제하고 loss를 convexify한다.

- **② Do MSAs act like Convs?**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/509ffa11-0d03-4909-9e2a-46b6d5101992)

  - MSA와 Conv는 서로 반대의 동작을 함
  - MSA는 feature map를 집계하지만 Conv는 feature map를 다양화함
  - 그림 2a: feature map의 Fourier analysis
    - MSA는 high-frequency signal를 감소시킴, Conv는 high-frequency signal를 증폭시킴
    - MSA는 low-poss filter, Conv는 high-pass filter
  - 그림 2b
    - Conv가 high-frequency noise에 취약하지만 MSA는 그렇지 않음
  - MSA는 Convs와 상호보완적

- **③ How can we harmonize MSAs with Convs?**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/895edece-a6d1-4443-a8a9-aea2bf1941e5)

  - multi-stage NN → 작은 개별 모델을 직렬로 연결, 직렬 연결처럼 동작
  - 각 stage의 마지막에 spatial smoothing를 적용하면 그림 3a와 같이 각 stage의 transformed feature map output를 조합하여  accuracy를 향상시킬 수 있음
    - 이런 결과를 바탕으로 convolution과 MSA를 번갈아 사용하는 패턴 제안
  - Conv와 MSA를 번갈아 사용하는 설계 패턴은 그림 3b와 같이 MLP 블록 하나 당 MSA 블록을 가지는 표준 transformer의 구조를 도출
  - 이런 설게 패턴은 대규모 dataset뿐만 아니라 CIFAR와 같은 소규모 dataset에서도 CNN보다 성능이 뛰어남
  - MSA가 단순히 기존 Conv를 대체하는 일반화된 Conv가 아닌 Conv를 보완하는 일반화된 spatial smoothing이라는 것을 의미

---

## 2. What properties of MSAs do we need to improve optimization?

- 저자들, MSA의 성질을 이해하기 위해 기본 ViT, PiT (ViT + multi-stage), Swin (ViT + multi-stage + local MSA)를 비교

  - 추가적인 inductive bias를 통해 ViT가 강력한 학습을 할 수 있음을 보여줌
  - $+$ ResNet도 비교

- NNs는 300epoch에 대해 DeiT-style data augmentation으로 처음부터 학습시킴

- **The stronger the inductive biaes, the stronger the representation (not regularizations)**

  - 질문! Weak inductive biases를 가진 모델이 학습 dataset에 overfitting 될까?

    ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/c5b3bbd2-1c8c-4bff-8116-237f6b80a779)

    - 위 질문의 답을 테스트 dataset의 오차와 학습 dataset의 cross-entropy 또는 negative log-likelihood ($NLL_{train}$, 낮을수록 좋음)를 기준으로 할 수 있다.
      - 실험결과, inductive bias가 강할수록 테스트 error와 train NLL이 모두 낮아짐
        - <span style='color:red'>예상과 다른 결과! weak inductive biases를 가진 모델은 train dataset에 과적합될 것으로 예상했지만, inductive bias가 강하니까 오히려 train dataset에서 fitting된다는 사실을 발견!</span>
        - ViT가 train dataset에 과적합되지 않음
      - MSA에 대한 locality constraints와 같은 적절한 inductive bias는 NN이 강력한 representation 학습에 도움을 줌

    ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/2da6f161-3739-40cc-b08f-01d656df2097)

    ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/59cc63d3-c4ed-4a7a-8809-bd9e02fd32cf)

    - 그림 C.2 → Weak inductive bias가 NN 학습을 방해하는 것을 보여줌<br/>해당 실험에서 patch 크기가 $2\times2$인 경우 ViT 예측 성능이 오히려 떨어짐

- <span style='color:red'>**★ 실험을 이해하기 위한 배경지식**</span>

  -  Test error and training NLL
    - NLL: train dataset에서 수렴을 평가하는데 적절한 지표, accuracy와 uncertainty을 모두 나타냄
      - 왜? NN은 NLL를 최적화하기 때문
    - Test error: 예측 성능 측정 위한 지표
    - 추가적인 inductive bias나 학습 기법이 NN의 성능을 향상시키는 경우 → NN이 "강력한 representation"을 학습하도록 돕는 방법 또는 "regularized"하는 방법 중 하나
      - 더 낮은 NLL은 이러한 bias나 기법이 강력한 representation 학습에 도움이 됨
      - 반대로, training NLL이 저하되면 (높으면) bias나 기법이 NN을 정규화한다는 것을 의미
      - Test error가 training NLL의 향상때문에 저하되면 NN이 training dataset에 과적합
  - Hessian max eigenvalue spectrum
    - Hessian max eigenvalue spectrum: real-world problem에 대해 거대 NN의 Hessian eigenvalue를 시각화하기 위한 방법
      - <span style='color:blue'>☆ Hessian matrix??</span>
        - <span style='color:blue'>함수의 이차미분을 나타냄 → 이차미분은 함수의 곡률 특성을 나타냄</span>
          - <span style='color:blue'>어떤 함수의 일차미분의 되는 점 critical point (극점), 이 극점이 극대인지 극소인지 변곡점인지 구분하기 위해 이차 미분값을 이용, 이때 사용되는것이 Hessian</span>
          - <span style='color:blue'>Hessian matrix → 대칭행렬, 고유값 (eigenvalue) 분해가 가능, 서로 수직인 n개의 고유벡터를 가짐</span>
            - <span style='color:blue'>eigenvalue가 모두 양수? 해당 지점에서 함수는 극솟값을 가짐 (아래로 볼록)</span>
            - <span style='color:blue'>eigenvalue가 모두 음수? 해당 지점에서 함수는 극댓값을 가짐 (위로 볼록)</span>
            - <span style='color:blue'>eigenvalue가 음과 양수로 섞인 경우: 해당 지점에서 함수는 **변곡점**을 가짐</span>
      - power iteration mini-batch 사용해서 top-k Hessian eigenvalue 계산 후 수집
      - 좋은 loss landscape? 평평하면서도 convex한 것
      - Hessian eigenvalue는 loss의 flatness (평평한 정도)와 convexity을 나타냄
        - Hessian eigenvalue 크기 → 선명도
        - negative Hessian eigenvector가 존재하면 non-convexity
      - 위의 insight를 바탕으로 Negative max eigenvalue proportion (NEP, 낮을수록 좋음)과 positive max eigenvalues의 평균 (APE, 낮을수록 좋음)을 도입해서 각각 비볼록성, 선명도를 정량적으로 측정 가능
        - NEP: 비볼록성, APE: 선명도
      - Hessian max eigenvalue spectrum $p(\lambda)$의 경우
        - NEP: $\int_{-\infty}^{0}p(\lambda)d\lambda$의 비율
        - APE: $\int_{0}^{\infty}\lambda p(\lambda)d\lambda / \int_{0}^{\infty}p(\lambda)d\lambda$의 expected value

- **ViT does not overfit small training datasets**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/bf44a92b-572e-4718-9eaf-2f6add2a9659)

  - 저자들은 ViT가 더 작은 dataset에서도 과적합되지 않다는 것을 관찰<br/><span style='color:red'>→ ??? weak inductive bias를 가지면 과적합된다고 예상했었음</span>
    - 실험을 통해 dataset의 크기가 감소합에 따라 예상대로 오차가 증가하지만, 놀랍게도 $NLL_{train}$도 같이 증가 
      - 왜? 강력한 data augmentation 덕분에 dataset의 크기가 2%인 경우에도 과적합하지 X <br/><span style='color:red'>→ 이게 의미하는 바?? 작은 dataset에서 ViT의 성능이 좋지 않은 원인이 과적합이 아니라는 것</span>

- **ViT's non-convex loss lead to poor performance**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/71d3b5a6-5aa4-4337-98d5-93f1bcda4f19)

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/3678947f-75d2-49ee-8dfa-b905a336bc91)

  - MSA의 weak inductive bias가 최적화를 어떻게 방해할까?
    - loss landscape관점에서 설명가능
      - ViT의 loss function은 non-convex, ResNet의 loss function은 (거의)convex
      - ViT의 loss function이 non-convex하기 때문에 훈련 초기 단계에서 NN 훈련을 방해
    - 그림 1b, 그림 4는 배치크기가 16, top-5 Hessian eigenvalue density를 보여줌
      - ViT가 negative hessian eigenvalue을 많이 가지고 있는 반면, ResNet은 몇 개만 가지고 있음
    - 그림 4는 대규모 dataset이 학습 초기 단계에서 Negative hessian eigenvalue을 억제한다는 사실을 보여줌
      - 대규모 dataset은 loss를 convex화해서 ViT가 강력한 representation을 학습하는데 도움이 되는 경향이 있음
      - ResNet은 작은 dataset에서도 loss가 convex하기 때문에 대규모 dataset의 이점을 거의 누리지 못함

- **Loss landscape smoothing methods aids in ViT training**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/2332ede0-8c7e-4300-9dc1-feb672478d6c)

  - loss landscape smoothing 방법은 ViT가 강력한 representation 학습에 도움을 줌
  - Classification task에서 GAP는 feature map point를 강력하게 조합하여 loss landscape를 smoothing함
  - 그림 6을 보면 GAP를 사용한 classifier가 negative hessian max eigenvalue을 억제하는 사실을 알 수 있고, 이는 GAP가 loss를 convex하게 만들어 준다는 것을 알 수 있음
  - Sharpness-Aware Minimization (SAM)도 NN이 smooth minima (최솟값)을 찾는데 도움을 줌 (다른 논문에서 SAM이 ViT 예측 성능을 향상시킴)

- **MSAs flatten the loss landscape**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/6f7bd586-bf18-4240-a0d7-cd6835dc0247)

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/3678947f-75d2-49ee-8dfa-b905a336bc91)

  - MSA의 또 다른 특성은 Hessian eigenvalue의 크기를 감소시킨다는 점
  - 위 그림들은 ViT eigenvalue가 CNN의 eigenvalue보다 훨씬 작다는 것을 보여줌
    - 무슨말??? Hessian eigenvalue는 loss function의 local curvation를 나타냄 <br/><span style='color:red'>→MSA가 loss landscape을 평탄화한다는 것을 보여줌</span>
  - 큰 eigenvalue는 NN학습을 방해  →  MSA는 큰 Hessian eigenvalue을 억제함으로써 NN이 더 나은 representation을 학습하는데 도움이 될 수 있음
  - MSA가 Hessian eigenvalue을 억제함으로써 loss landscape 평탄화한다는 것을 보여줌 (그림 1a)
  - 대용량 dataset에서는 MSA의 단점인 negative Hessian eigenvalue가 사라지고 장점만 남게됨 <span style='color:red'>→대용량 dataset에서는 ViT가 CNN을 능가함</span>

- **A key feature of MSAs is data specificity (not long-range dependency)**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/b88aba6c-b92c-4c31-808c-f4d32afae7c9)

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/2332ede0-8c7e-4300-9dc1-feb672478d6c)

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/0e10ae51-ab06-4186-b940-60b06206abff)

  - MSA의 2가지 특성
    1. long-range dependency
    2. data specificity (data dependency)
  - <span style='color:red'>**일반적인 믿음과 달리**</span>, long-range dependency는 NN 최적화를 방해
    - 이를 입증하기 위해 global MSA 대신 Convolution MSA로 구성된 Convolution ViT를 분석
    - Vision task를 위한 Convolution MSA는 2차원 convolution과 동일한 방식으로 feature map 전개 후 convolution receptive field의 feature map point 사이에서만 self-attention을 수행
    - 그림 7a는 kernel 크기가 $3\times3$, $5\times5$, $8\times8$ (global MSA)인 Convolution ViT의 error와 $NLL_{train}$을 CIFAR-100에서 보여줌 <span style='color:red'>→ $5\times5$ kernel이 가장 좋음</span>
  - 그림 7b는 강력한 locality inductive bias가 계산 복잡도를 감소시킬 뿐만 아니라 loss landscape을 convex화해서 최적화에 도움이 된다는 것을 보여준다.
    - $5\times5$ kernel은 불필요한 자유도를 제한 → negative eigenvalue가 더 적음
    - $5\times5$ kernel은 $3\times3$ kernel보다 더 많은 수의 feature map point를 조합하기 때문에 $3\times3$ kernel보다 negative eigenvalue가 더 적음
    - <span style='color:red'>이 두가지 효과가 균형을 이루면 negative eigenvalue의 양의 최소화됨 (그림 6에서 확인가능) <br/>무슨말?? 그림 6에서 GAP로 feature map point를 조합해서 loss landscape를 smoothing <br/>→ negative hessian eigenvalue ↓</span>
  - **결론, Global MSA로 long-range dependency를 포착안해도 됨 → data specificity가 NN을 향상시킴**
  - $+$ 그림 C.3에서 확인할 수 있듯이 small dataset에서는 epoch를 늘려도 ViT 예측성능이 ResNet 보다 안좋다.

- **A smaller patch size dose not always imply better results**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/77e24940-5983-4ad6-ac80-de355555c3f4)

  - patch 크기 ↓, representation 유연성 ↑, inductive bias ↓
  - 적절한 patch 크기는 ViT가 강력한 representation을 학습하는데 도움을 주지만 ViT를 정규화하지 X
    - 왜? 그림 C.2b에 따르면 patch 크기가 작을 때 hessian eigenvalue 크기는 감소하지만 negative hessian eigenvalue가 생성됨 <span style='color:red'>→ weak inductive bias로 인해 loss landscape가 평평하지만 convex하지 않은 형태가 됨</span>
    - patch 크기가 크면 negative eigenvalue가 억제되지만, 모델의 표현을 제한할 뿐만 아니라 loss landscape도 뽀족해짐

- **A multi-stage architecture in PiT and a local MSA in Swin also flatten loss landscape**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/66870c3e-452c-4da8-9dad-27b9b2cb9e4a)

  - 학습 초기 단계에서 PiT의 multi-stage architecture는  negative hessian eigenvalue를 억제하여 학습에 도움을 줌 (그림 C.4b)
  - Swin의 local MSA는 negative eigenvalue를 생성하지만 eigenvalue 크기를 크게 줄임 (그림 C.4b)

- **A lack of heads may lead to non-convex losses**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/74ad77a3-48e0-4507-97ba-0cf682bba4b2)

  - MSA에서 head가 충분하지 않으면 non-convex하고 급격한 loss가 발생할 수 있음
  - head 수가 증가함에 따라 NEP, APE 모두 감소하는 것을 그림 C.5에서 확인 가능
  - head 당 embedding dimension이 높을수록 loss가 convex해지고 평탄화되는 것을 보여줌 (그림 C.6)

- **Large models have a flat loss in the early phase of training**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/69d499b5-5eb4-4bbd-9704-6863f3148753)

  - 모델 크기 ↑ loss가 더 sharp 해짐

---

## 3. Do MSAs Act Like Convs?

- Convs는 data 정보를 활용하지 않고 채널 정보를 혼합한다는 점에서 데이터에 구애받지 않고 채널에 특화되어있다.
- 반대로, MSAs는 data에 특화되어있고, 채널에 구애받지 않는다.
  - <span style='color:red'>Convs와 MSA는 상호보완적인 관계</span>

![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/4e52bf6c-86e2-4dbe-ad3e-3c8a7a871573)

- **MSAs are low-pass filters, but Convs are high-pass filters**

  - MSA는 self-attention importance가 잇는 feature map를 공간적으로 부드럽게 만드는 효과가 있음
    - <span style='color:red'>MSA는 high-frequency signal를 줄이는 경향이 있을 것으로 예상</span>
  - 그림 8, ImageNet 기준, high-frequency ($1.0\pi$)에서 ViT의 fourier transformed feature map의 relative log 진폭 ($\bigtriangleup log$ 진폭)을 보여줌
    - 해당 그림을 보면 MSA가 항상 high-frequency를 감소시킴!! <span style='color:red'>(model 초기 단계 제외)</span>
    - 모델 초기 단계에서는 진폭을 증가시키기 때문에 MSA가 Conv처럼 동작
    - <span style='color:red'>어떤 insight를 얻을 수 있을까? <br/>초기 단계에서는 Conv를 사용하고 그 이후 단계에서는 MSA를 사용하는 hybrid 모델에 대한 근거!</span>

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/25fb9bd4-da6b-422f-af7c-97e14b9b700b)

  - 그림 D.2에서의 흰색 영역: Conv / MLP, 회색 영역: MSA, 파란색 영역: Sub-sampling (Maxpooling 같은 것)
  - low-frequency 신호와 high-frequency 신호가 각각 MSA와 Convs에 유익한 정보를 제공한다는 사실을 알 수 있음
    - <span style='color:red'>실제로 실험해보면 ViT는 low-frequency 잡음에 취약하고 ResNet은 high-frequency 잡음에 취약하다는 것을 알 수 있다.</span>
    - low-frequecy signal: 이미지의 shape
    - high-frequency signal: 이미지의 texture
    - <span style='color:red'>결론? MSA는 shape-biased (모양 편향적), Conv는 texture-biased (질감 편향적)</span>

- **MSAs aggregate feature maps, but Convs do not.**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/677dae7a-4191-4a78-ac48-43dcd3bef7a8)

  - MSA는 feature map 평균화 → feature map points의 분산을 줄일 수 있음. <span style='color:red'>→ MSA가 feature map을 ensemble한다는 것을 의미</span>
  - 그림 9에서 ViT의 MSA (회색영역)가 분산을 감소시키는 경향이 있고, ResNet의 Conv와 ViT의 MLP (흰색 영역)는 분산을 증가시키는 경향을 보여줌
  - MSA는 feature map을 ensemble하지만 Conv는 그렇지 X<br/><span style='color:red'>→ feature map uncertainty을 줄이면 transformed feature map를 ensemble하고 안정화해서 최적화에 도움을 줌</span>

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/4b26991d-34d3-4265-b4ab-fe70634c7a40)

  - feature map 분산은 2가지 패턴을 보임
    1. 분산은 모든 NN layer에 누적되며 깊이가 증가함에 따라 증가하는 경향이 있음
    2. ResNet의 feature map 분산은 각 단계의 끝에서 정점에 도달
  - ResNet의 경우 각 단계의 마지막에 MSA를 추가하면 성능이 향상됨
    - <span style='color:red'>왜? 마지막에 분산을 줄이는 역할을 MSA가 해주기 때문이다.</span>
  - 그림 D.1은 PiT와 Swin의 MSA도 feature map 분산를 줄여주는 것을 보여줌
    - $+$ MSA는 단계의 시작 부분에서 분산을 억제하지만 단계의 끝부분에서는 그렇지 X

- **★ Multi-stage ViTs have block structure**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/923368cc-7a10-4ab7-a115-2af1b96d9dfc)

  - CNN의 feature map 유사성은 block 구조로 나타남
  - ViT는 모든 layer에 걸쳐 균일한 representation을 가짐
  - 그림 D.3를 보면 multi-stage의 sub-sampling (pooling) layer가 ViT의 feature map representation을 block구조로 만들어 주는 것을 알 수 있다.

- **★ Convs at the beginning of a stage and MSAs at the end of a stage play an important role.**

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/cdf901f4-736a-43f4-93b1-368c08fc2cac)

  - 그림 D.4는 ResNet과 ViT에 대한 <span style='color:blue'>"resion study"</span>결과를 보여줌

    - <span style='color:blue'>resion study?<br/>뇌 또는 신경계통에서 생긴 손상이나 병변을 연구하여 그 기능을 이해하는것을 의미 → 뇌의 특정 영역이 특정 기능을 당담하는지, 그리고 어떤 기능에 영향을 주는지 알아내기 위해 사용됨</span>

    - <span style='color:blue'>Deep learning에서의 resion study?<br/>일부 neuron 또는 neuron group을 제거하거나 기능을 변화시켜보는 것을 의미<br/>→ 이를 통해 그 영역이 모델의 성능에 어떤 영향을 미치는지 파악할 수 있다. 특정 neuron을 비활성화시키거나 neuron의 가중치를 변경하면 모델의 출력이 어떻게 변하는지 관찰 가능<br/>→ 각 neuron 또는 neuron group의 기능과 모델의 예측 능력 사이의 상관관계를 "resion study"를 통해 이해하고자 함</span>

  - 이 실험에서는 ResNet의 bottleneck block에서 $3\times3$ layer 하나, ViT에서 MSA 또는 MLP 블록 하나를 제거함

    - stage의 시작부분에서 Conv 제거하고 stage 마지막 부분에서 MSA를 제거하면 정확도가 크게 저하됨

---

## 4. How Can We Harmonize MSAs With Convs?

![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/be30bb49-d1e1-4a7f-a807-c47747e457c7)

- MSA와  Conv가 상호보완적이라는 특징을 활용, 그림 3c 설계 제안
- 대용량 dataset과 작은 dataset (CIFAR)에서 모두 CNN보다 성능이 뛰어남

### 4.1 Designing Architecture

![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/7d3d770e-1d1e-4e3f-b502-f1db69e3b0a2)

- ViT에 multi-stage 구조 적용하면 CNN처럼 feature map 유사성이 block 구조를 가지게 됨

- ResNet의 bottleneck에서 $3\times3$ Conv layer 하나 제거, Swin에서는 MSA 또는 MLP 블록 제거

  - ResNet에서 초기 단계의 layer를 제거하면 그 후 단계의 layer를 제거하는 것보다 큰 영향을 끼침
  - Swin에서 stage 시작시 MLP제거하면 정확도 감소, stage 마지막에 MSA제거하면 정확도가 심각하게 감소
  - <span style='color:red'>Conv와 MSA 조합시 Conv는 앞부분에 MSA는 마지막 부분에 두는 것이 가장 좋다!<br/>MSA는 stage 마지막에 가까워질수록 성능이 크게 향상</span>

- 저자들이 모델 구조 설계할 때 세운 규칙

  - 기존 CNN model 끝에서 Conv block를 MSA block으로 대체
  - 추가된 MSA block으로 성능 향상 안되면 이전 stage 끝에 위치한 Conv block을 MSA block으로 대체
  - 후기 단계의 MSA block에는 더 많은 head와 더 높은 dimension 사용

  ![image](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/36d0015e-2f84-4c42-afcd-4df518cf6b8c)

  #### 결론! Multi-head Self-Attention은 Convolution을 보완하는 일반화된 Spatial smoothing!!!

 <br/>

> **Reference** 
>
> https://arxiv.org/abs/2202.06709
