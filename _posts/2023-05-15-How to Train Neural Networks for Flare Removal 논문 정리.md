---
key: jekyll-text-theme
title: How to Train Neural Networks for Flare Removal 논문 정리
excerpt: 'How to Train Neural Networks for Flare Removal 논문 읽고 정리하기'
tags: [AI, Computer Vision, Paper, CNN, Lens flare, Denoising]
---

# How to Train Neural Networks for Flare Removal (ICCV, 2021)

## Abstract

- 카메라가 강한 광원을 가리킬 때, 사진은 lens flare artifact가 만들어진다.
  
  - 다양한 패턴: halos(후광), streaks(줄무늬), color bleeding(색 번짐), haze(안개?, 흐릿함?) 등
  
  ![Untitled](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/5b91edfd-fdfa-440f-b585-76bb0d20ff78)
- 현존하는 분석 방법은 artifact의 형태, 밝기에 대해서 분석을 하기 때문에, 일부 flare의 subset에 대해서만 잘 분석한다.

- Machine Learning 기술들은 reflection(반사)와 같은 artifact와 같은 유형을 제거하는데 성공했지만 훈련 데이터가 부족해서 광범위하게 flare removal를 적용할 수 없다.

  - 이 논문에서는 이런 문제를 해결하기 위해서, 경험적으로 또는 wave optics(파동 광학)를 사용해서 flare의 광학적 원인을 명시적으로 모델링하고 flare로 손상된 이미지와 깨끗한 이미지의 semi-synthetic 쌍들을 생성.
    - 이는 lens flare를 제거하도록 neural network들을 훈련할 수 있도록 해주었음

- 이 논문의 실험에서 data synthesis 방법이 flare removal task에 중요한 영향을 끼치는 것을 확인했고, 논문에서 제안한 data synthesis 방법으로 학습된 model들은 다양한 장면, 조명 조건 및 카메라에 걸쳐 실제 lens flare에 대해서도 일반화를 잘하는 것을 확인할 수 있었다.


---

## Introduction

- 강한 광원이 있는 사진은 의도하지 않은 반사, 그리고 카메라 내부에서 산란(scattering)에 의해 시각적으로 두드러진 lens flare artifact가 나타난다.
  - Flare artifact들은 이미지를 망가트리고, detail을 줄이고, 이미지의 내용을 가릴 수 있음.
- lens flare를 줄이기 위한 광학적(optical) 설계에도 여전히 flare가 발생
  - 심지어 작은 광원에도 여전히 상당한 artifact들을 만들어진다.
- Flare artifact들은 lens의 광학(optics), 광원의 위치, 카메라 제조 상의 결함 및 매일 사용하면서 누적된 scratch들, 먼지(dust)에 의해 발생
  - lens flare는 다양한 원인에 따라 다양한 형태로 존재
    - halos, streaks, bright lines, saturated blobs, color bleeding, haze 등
- 현존하는 lens flare removal 방법들은 flare 형태에 대한 물리학을 고려하지 않지만, 오히려 artifact를 식별하고 localize하기 위해 template matching 또는 intensity thresholding에 의존한다.
  - 이 방법들은 탐지만 하거나, saturated blobs와 같은 한정된 flare 유형만 제거하며, 더 복잡한 실 생활의 시나리오에서는 잘 적용되지 않음

### 문제?

- **훈련 데이터 부족**

  - 완벽하게 lens flare가 있는 이미지와 lens flare가 없는 이미지의 쌍을 모으기 힘들다.
    사진 작가가 직접 찍어서 모은 이미지는 엄청난 노동력이 들어가고 flare를 유발하는 광원이 카메라의 시야 밖에 있을 때만 적용되므로 유용성이 제한된다.

  - 이런 문제를 해결하기 위해서, 논문 저자들은 물리학 법칙에 바탕을 둔 **semi-synthetic data**를 생성하는 것을 제안

  - 논문 저자들은 lens flare가 이미지 내 추가적인 상단 layer에 존재하고, 이는 산란 또는 내부 반사에 의해 발생한다는 것을 관찰했다.

    - 카메라로 찍은 장면, 풍경에 카메라 내부에서 일어나는 현상으로 인해 flare가 합쳐져서 이미지 결과물이 나옴
      flare를 합성할 때, 이미지 장면 내에서 광원이 있을만한 위치에 합성!

    ![Untitled 1](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/e276777b-78bb-4f6e-9f2c-3426e646b1ce)

    ![Untitled 2](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/e4a1c78a-681d-4b20-9c96-e0299e10306d)

  - 산란(Scatterting)

    - Scratch 및 dust 등 다른 결점들에 대해 논문 저자들은 실제와 매우 근접하게 설명되는 wave optics model(파동 광학 모델)을 구축
    - lens의 요소들 사이에서 의도치 않은 반사에 대해, 저자들은 상업용 카메라에 대한 정확한 광학 모델을 사용할 수 없는 경우가 많기 때문에 엄격한 데이터 기반(data-driven) 접근 방식을 채택
      - 이런 방식은 Ground-truth flare-free 이미지들과 짝을 이루는 Semi-synthetic flare-corrupted의 크고 다양한 데이터셋을 생성할 수 있음
      - 이런 방식은 Ground-truth flare-free 이미지들과 짝을 이루는 Semi-synthetic flare-corrupted의 크고 다양한 데이터셋을 생성할 수 있음

- **눈에 보이는 광원을 손대지 않고 유지한 채 flare를 제거하지 못하는 문제**

  - 이런 문제는 논문에서 제안한 Semi-synthetic data로도 해결하기 어려운 문제
    - 왜?
      광원에 의해 생긴 flare를 flare-only layer로 부터 광원을 분리 시킬 수 없었기 때문이다.
      즉, 광원과 flare는 완전 다른 것!
    - flare라는 것은 광원으로부터 나오는 부산물, 광원은 원래 있는 자연스러운 것, flare는 광원에 의해 발생한 부자연스러운 결과
  - 위 문제를 고려하지 않은 채 network를 훈련한다면, network는 flare와 함께 광원을 제거하려 할 것이고, 이는 비 현실적인 결과물로 도출됨
    - 왜? 
      위에서 나온 것처럼, 광원과 flare를 분리 시킬 수 없었기 때문
  - 그래서 논문 저자들은 **광원 부분을 무시하는 loss function 및 출력에서 광원을 보존하기 위한 후 처리 과정을 제안**

- 논문에서 제안한 데이터셋과 그 과정의 효과를 보여주기 위해, 다른 task를 위해 고안된 2개의 CNN을 훈련시킴

- 훈련하는 동안, 논문 저자들은 예측된 flare-free 이미지와 residual(즉, inferred flare(추론된 flare)) 둘 모두에 대해 loss function을 최소화 시킴

- test 할 때, network는 기본 카메라로 찍은 single RGB 이미지만 요구되고 다양한 장면에 대한 flare의 다른 유형을 제거할 수 있다.

- 오로지 Semi-synthetic data로만 훈련했지만, 두 model들은 실 생활 이미지들에 대해 일반화를 잘한다.

---

## Related works

- flare removal 해결법 3가지
  - flare 존재를 완화 시키기 위해 의도한 광학 설계
  - 찍은 후 이미지 향상을 시도하는 Software-only 방법
  - 추가적인 정보를 잡아내는 Hardware-Software 해결법

### Related work - Hardware Solution

- 최상급 카메라의 렌즈들은 정교한 광학적 설계와 재료들을 채택
- 각 glass element로 이미지 quality를 향상 시키기 위한 합성 lens를 추가함에 따라, 빛이 lens 표면으로부터 반사되어 flare가 생성될 확률이 증가!
  - 이를 방지하기 위해 널리 사용되는 기법 중 하나는 lens 요소들에 anti-reflective 코팅 (AR 코팅)을 적용하는 것이고, 이는 destructive interference로 인해 내부 반사가 줄어든다.
  - 그러나, 이 코팅은 특정 파장과 투사(Projection)의 각도에서만 최적화되어 있어 완벽하지는 않음
    추가적으로, 모든 광학(optical) 표면에 AR 코팅을 하는 것 비싸고, 이런 코팅은 다른 코팅(anti-scratch, anti-fingerprint)에 의해 제외되거나, 간섭 받을 수 있음

### Related work - Computational methods

- 많은 후 처리 기술들이 flare를 제거하기 위해 제안
- Deconvolution은 X-ray imaging 또는 HDR 사진에서 flare를 제거하기 위해 사용되어왔음
  - 이런 방법들은 flare의 point spread function(PSF)(?)이 공간적으로 다양하지 않다는 가정에 의존한다.
    그리고 이런 가정은 일반적으로 사실이 아니다.
- 다른 방법들은 2단계 절차를 채택
  - lens flare의 독특한 모양, 위치, 또는 강도(Saturation region을 식별함으로써)를 기반으로 lens flare를 탐지하고 inpainting(복원)을 사용하여 flare 영역 뒤의 장면을 복원
    - 이러한 방식들은 flare의 한정된 유형(예, bright spots)에만 적용되고, flare로서 모든 bright region들의 misclassification에 취약하다.
    - 추가적으로, 이런 기술들은 대부분 lens flare들이 장면 안에서 상단에 semi-transparent layer로서 더 잘 만들어진다는 사실을 무시하고 각 pixel을 “flare” 또는 “not flare”로 분류한다.

### Related work - Hardware-Software Co-design

- 연구자들은 카메라 하드웨어와 후 처리 알고리즘이 같이 설계된 flare removal을 위한 computational imaging을 사용하곤 했음
  - Talvala et al, Raskar et al은 구조화된 occlusion(폐색) mask들을 사용해 선택적으로 flare를 일으키는 빛을 막으려 시도했고 direct-indirect separation 또는 light field 기반 알고리즘 둘 다 사용해서 flare-free scene으로 복원하려 시도했음
  - 이런 방법들은 훌륭하지만 특별한 하드웨어를 요구하므로 실용성이 떨어진다.

### Related work - Learning-based image decomposition

- no learning-based flare removal 기술이 존재하는 반면에, 최근 여러 연구에서 reflection removal, rain removal 그리고 denosing 과 같은 유사한 응용 분야에 learning(학습)을 적용했다.
  - 이런 방법들은 neural network를 훈련 시킴으로서 이미지를 “clean” 그리고 “corrupt” 구성 성분들로 분해를 시도한다.
  - 이 방법들이 성공하려면 high-quality domain-specific training-dataset이 필요하다.

---

## Physics of lens flare

- 초점이 맞을 때, 이상적인 카메라는 광원이 있는 지점으로부터 센서의 어떤 하나의 지점으로 모든 광선이 굴절되어 수렴한다고 가정한다.

  - 실제로, lens는 의도치 않은 경로를 따라 빛이 산란하고 반사되어 다음 사진에서 보이는 것처럼 flare artifact들이 나오는 것을 볼 수 있다.

  ![Untitled 3](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/1b72ca26-6b71-4ac4-b140-a4123fe5b9b1)

  - 산란, 반사된 부분들은 각각 입사 광선의 작은 파편으로 구성
    - 그래서 flare가 편재하더라도, 대부분 사진들에서 눈에 띄지 않는다.
    - 그러나, 강한 광원이 장면의 나머지 부분보다 훨씬 더 밝을 때 이 밝은 빛에서 산란 및 반사된 광선의 작은 부분이 이미지의 다른 픽셀에서 눈에 보이는 artifact를 초래
  - dust와 scratch들로 인한 산란의 형태, 다중 반사의 형태는 특정하게 눈에 보이는 패턴들로 나타남
  - flare는 2가지 주된 범주로 분류할 수 있음
    - 산란에 의해 유발된 flare
    - 반사에 의해 유발된 flare

### Physics of lens flare - scattering flare

- 이상적인 lens는 100% 굴절되는 반면, 실제 lens는 많은 결점에 의해 빛이 산란한다.
- 산란(또는 회절)은 제조 상 결점(예, 찌그러짐) 또는 정상적인 마모(예, dust(먼지) 그리고 scratch들) 때문에 발생
- 결과적으로 굴절된 주 광선으로부터 떨어져 나간 두 번째 광선들의 집합이 의도한 경로 대신에 산란하거나 회절된다.

![Untitled 4](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/dc355c05-427f-4f71-918a-733d994a3286)

- dust는 무지개와 같은 효과를 얻는 반면, scratch들은 광원으로부터 빠르게 “방출”되면서 streaks가 나타난다.
- 산란은 광원 주변 부분에서 대비를 감소 시켜서 hazy(흐릿한) 상태가 나타난다.

### Physics of lens flare - reflective flare

- 실제 lens 시스템에서, 각 공기와 유리 사이의 상호작용에서 반사될 확률이 4%라 주장
- 짝수 번의 반사 후에 광선은 반사 패턴을 형성하면서 센서의 의도하지 않은 위치에 닿는다.
- 빛이 정확히 두 번 반사한다고 가정해도, n개의 광학적(optical) 요소들을 포함하는 lens 모듈의 경우(현대의 카메라의 경우 n ≒ 5) 2n개의 광학(optical) 표면이 있으므로(왜? 반사가 되니까 x2) n(2n-1)개의 잠재적인 flare 유발 조합이 존재
  - 반사가 2번보다 많이 된다고 하면, 더 많은 잠재적인 flare 유발 조합이 존재
- 이미지에서 reflective flare들은 광원과 주요한 지점에 합쳐지는 직선 line에 놓여있음
  - reflective flare들은 광원의 입사각에 민감하지만 광학(optical)축에 대한 회전에 대해서는 민감하지 않음
- reflective flare의 모양은 형태, 크기 그리고 한 번 이상 반사를 부분적으로 막아주는 조리개의 위치에 의존
- AR 코팅 → 공기와 유리 사이의 상호작용 중 반사 확률을 1%이하로 줄여줌
  - AR 코팅의 효과는 파장에 의존, 따라서 lens flare가 나타나면 다양한 색조로 나타남
    (파랑, 보라 또는 핑크)
- reflective flare는 lens 설계에 의존
  - 같은 카메라면 비슷한 장면에서 유사한 reflective flare 발생

### Physics of lens flare - challenge in flare removal

- flare의 다른 유형들은 종종 시각적으로 구별하거나 분리하기 힘듬
- 관찰된 flare의 형태는 광원의 특성(예, 위치, 크기, 강도, 그리고 스펙트럼) 및 lens(예, 설계 그리고 결점)에 기반해서 상당히 다양함
  - 이런 이유로 인해 flare를 분석적으로 식별하고 flare의 각 유형을 제거하기 위한 물리학 기반 알고리즘을 완벽하게 구현하는 것은 비현실적이다.
    특히, 여러 개의 artifact들이 같은 이미지에 나타날 때 구현하기 힘들다.
  - 그러므로, 논문 저자들은 데이터에 기반한 접근법을 제안

---

## Physics-based data generation

- 지도 학습 환경에서 많은 비전 문제들과 다르게, 이 flare removal은 flare-corrupted와 flare-free 이미지 쌍의 데이터셋을 얻기 어렵다.
- 빛의 더하기 성질은 저자들이 “이상적인”광선 강도의 감소가 무시될 수 있는 “이상적인” 이미지의 위에 첨가하는 artifact로서 flare를 모델링할 수 있다는 것을 암시 (무슨 의미?)
  - 이를 이용해서 데이터를 생성할 수 있음

### Physics-based data generation - Scattering flare

- **Formulation**

  - 얇은 lens의 방면에서, optical imaging 시스템은 복소수 pupil function P(u, v)에 의해 특정할 수 있다.
    pupil function P함수: 조리개 평면에서 각 지점 (u, v)에 대한 진폭과 파장 λ를 가진 입사파의 위상에 대해 lens의 효과를 설명하는 2D field

      $P_λ(u, v)=A(u, v)ㆍexp(iΦ_λ(u, v))$  (1)

      - (u, v)에서의 위상에 대한 lens 효과 설명하는 2D field

  - A는 입사파 진폭의 감쇠를 표현하는 광학 성질에 대한 apreture function

    $A(u, v)={\begin{cases} 1 \quad if\, u^2+v^2 <r^2 \\ 0 \quad otherwise \end{cases}}$ (2)

    r : 카메라 aperture(조리개) 반지름

    - (u, v)가 조리개 반경 내에 존재하면 1 아니면 0

  - 식 (1) $Φ_λ$ 
    → 광원의 3D 위치 (x, y, z)  뿐만 아니라 파장에 의존하는 phase shift(위상 변화) 설명
    $Φ_λ(x, y, z) = {Φ_λ}^s(x/z, y/z) + {Φ_λ}^{DF}(z)$ (3)

      - ${Φ^s}$는 입사각에 의해 설명됨
      - ${Φ^{DF}}$는 광원 지점의 깊이 z에 의존 (defocus)

  - 식 (1) pupil function P는 Fourier 변환에 의한 point spread function(PSF)를 사용해 계산 가능
    $PSF_λ =|F\lbrace P_\lambda \rbrace|^2$  (4)

      - 이는 정의 상 aperture function A가 있는 ((u, v)가 조리개 반경 r 내에 있다.) 카메라에 의해 형성된 (x, y, z) 지점의 광원 이미지이다.
        → 이게 논문 저자들이 원하는 flare 이미지

- **Sampling PSFs (논문 부록에 PSF 보고 정리하기)**

  - lens  위 dust와 scratch들을 모방하기 위해, aperture function A 식 (2)에 무작위 크기와 투명도의 점들과 줄무늬(streaks)를 추가
  - 하나의 파장 λ를 가진 (x, y, z) 위치에서 광원이 주어지면, 식 (3)의 ${Φ^s}$, ${Φ^{DF}}$ 2개의 phase shift 항들을 계산 가능
  - 광원에 대한 $PSF_λ$는 A와 $Φ_λ$ 를 식 (4)에 대입하여 풀 수 있음
    ${\begin{pmatrix}
    PSF_R(s, t) \\
    PSF_G(s, t) \\
    PSF_B(s, t)
    \end{pmatrix}} = SRF{\begin{pmatrix}
    PSF_{λ=380nm}(s, t) \\
    \vdots \\
    PSF_{λ=740nm}(s, t)
    \end{pmatrix}}$ (5)
  - 전체 가시광 스펙트럼에 걸쳐 광원을 시물레이션 하기 위해, 전체 파장 380nm ~ 740nm 까지 5nm씩 나눠서 PSF의 각 pixel 당 73개의 vector가 되도록 추출
  - full-spectrum PSF에 RGB 센서가 측정한 PSF에 근거한 spectral response function SRF(3 X 73 행렬)를 왼쪽으로 곱한다.
  - (s, t) → 이미지 좌표

  → 이는 (x, y, z)에 위치한 광원에 대해 flare 이미지를 만든다.

  - Scattering flare들에 대한 데이터셋을 만들기 위해 aperture function A 및 spectral response function SRF를 무작위로 추출
  - $PSF_{RGB}$ 이미지들을 augumentation 하기 위해 광학적 왜곡(예, barrel, pincushion)을 적용
    → 총 2000개의 이미지 생성
      - barrel: 줌 렌즈의 초점 길이가 가장 짧을 때 발생 (최대 광각점)
      - pincushion: 가장 긴 초점 길이일 때 발생

### Physics-based data generation - Reflective flare

- reflective flare 구성 성분은 rendering 기술로 구현하기 힘들다.
  - 왜? 얻을 수 없는 정확한 광학 특성을 요구하기 때문
  - 하지만, 유사한 설계의 lens는 비슷한 내부 반사 경로를 공유한다, 그래서 카메라의 하나의 instance에서 수집된 데이터는 다른 유사한 instance들에도 일반화를 잘 한다.
- 저자들은 밝은 광원, 프로그래밍이 가능한 회전 stage, 그리고 f=13mm lens를 가진 aperture(조리개)가 고정된 스마트폰 카메라로 구성된 실험실 환경에서 reflective flare 이미지를 찍음
  - 카메라 → $-75^{\circ}$ ~ $75^{\circ}$로 광원을 대각선으로 추적하면서 회전
    - $0.15^{\circ}$ 마다 하나의 HDR 이미지를 캡처해 1000개의 이미지 생성
    - 인접한 캡처들은 [Simon Niklaus and Feng Liu. Context-aware synthesis for video frame interpolation. CVPR, 2018]의 frame interpolation 알고리즘을 사용해 2배로 보간되어 2000개의 이미지들을 얻음
    - 훈련하는 동안, reflective flare는 광학 축을 중심으로 회전 대칭을 하므로, 이미지를 무작위로 회전 시켜 더 augumentation 시킴

### Physics-based data generation - Synthesizing flare-corrupted images

- flare-corrupted 이미지 $I_F$는 선형 공간에서 flare-only 이미지 F를 flare-free natural 이미지 $I_O$에 더함으로서 생성

  - linear space: pixel 강도가 추가된 사전 색조가 매핑된 원시 공간

- random gaussian noise 추가

  - gaussian noise의 분산은 scaled chi-sqaure 분포 $σ^2$ ~ $0.01χ^2$로부터 이미지 당 한 번 sampling 된다.
    - 왜? 사람들이 볼 것으로 예상되는 noise 수준을 포함하기 위해

  $I_F = I_O + F + N(0, σ^2)$ (6)

  - flare 이미지와 원본 이미지 합성 시 가우시안 노이즈 합치는 것 생각

![Untitled 5](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/76b8f453-62de-43a2-80d1-4c615f2bf55a)

- flare-free 이미지 $I_O$는 무작위 반전 그리고 밝기 조절로 augmentation
  - 저자들이 사용한 flare-free 이미지는 gamma가 인코딩(부호화)되어있음.
    그래서 정확한 값을 알 수 없다는 사실을 설명하기 위해 [1.8, 2.2]에서 균일하게(Uniformly) $\gamma$를 sampling하는 역 감마 곡선을 적용하여 대략적으로 선형화
- flare-only 이미지 F는 captured와 simulated 데이터셋 양 쪽에서 뽑아옴
  - captured와 simulated가 필요한 이유는 **Results - Ablation study** 부분에서 나온다.
  - 무작위 affine 변환 (예, scaling, translation, rotation, 그리고 shear) 그리고 white blance로 추가적인 augmentation 해줌

---

## Method

### Reconstruction algorithm

- 주어진 flare-corrupted 이미지 $I_F \in [0, 1]^{512 \times 512 \times 3}$에서, 저자들의 목적은 flare-free 이미지 $I_O$를 예측하기 위해 neural network $f(I_F, Θ)$를 훈련 시키는 것
  - $Θ$는 훈련 가능한 network weight
  
- flare removal task에 적합한 많은 network 구조 중 2개 사용해 평가


### Reconstruction algorithm - Losses

- 밝은 광원에 의해 생기는 flare만 제거하기를 원한다.

  - 불가능! 왜?
    데이터셋 안에는 capture하거나 simulatation 하는 동안 물리적으로 분리가 불가능한 flare와 광원이 flare-only 이미지 F에 포함되어있기 때문

- network를 순수하게 훈련 시키면, 광원이 제거된 이미지를 hallucinate하려 시도할 것이고, 이는 이 model의 의도 된 사용이 아니고 model 용량을 낭비하는 것

- 광원 뒤 장면을 model이 복원하는 것을 막기 위해, loss를 계산하고 광원 때문이라고 가정할 때 saturated pixel들을 무시 해야함

  - 일반적으로, saturated pixel들은 복원이 불가능하고 장면에 대한 아주 적은 정보만 가지고 있음.

- loss를 계산하기 전에 binary saturation mask M을 가진 network output $f(I_F, Θ)$를 수정함

  - mask M은 input $I_F$의 휘도(luminance)가 임계값보다 훨씬 큰 pixel들과 일치
    (실험에서 0.99, 0.99는 임계값)
  - morphological(형태학적) 연산을 적용하고 작은 saturated 영역들은 M에서부터 제외시킴
    이 영역들은 장면의 일부 또는 flare의 일부이다.

- M 안의 pixel들에 대해, ground truth $I_O$로 pixel들을 대체시킴, 그래서 몇몇 영역에 대한 loss는 0이다.
  $\widehat{I_O}=I_O \bigodot M + f(I_F, Θ) \bigodot (1-M)$

    - $\bigodot$ → element-wise multiplication

- 훈련하는 동안, 2개 loss들(image loss와 flare loss) 합을 최소화

  $L = L_I + L_F$

- image loss ($L_I$)는 예측된 flare-free 이미지 $\widehat{I_O}$가 photometrically(측광)및 지각적으로 모두 ground truth $I_O$와 유사해지도록 함

  - data term은 $\widehat{I_O}$와 $I_O$사이의 RGB 값들에 대한 L1 loss 이다. (더 공부)
  - perceptual term은 pre-trained VGG-19 network에 $\widehat{I_O}$와 $I_O$를 줌으로써 계산됨 (더 공부)
  - 선택된 feature layer (conv1_2, conv2_2, conv3_2, conv4_2 그리고 conv5_2) 에서 $Φ_l(\widehat{I_O})$와 $Φ_l(I_O)$ 사이의 절대 차를 패널티로 준다.
    $L_I = ||\widehat{I_O} - I_O||_1 + \sum_{l}{\lambda_l}||Φ_l(\widehat{I_O}) - Φ_l(I_O)||_1$ (9)

- flare loss ($L_F$)는 예측된 flare가 ground-truth flare F와 유사해지도록 하고, 예측된 flare-free 이미지에서 artifact들이 줄어들도록 쓰인다.
  $\widehat{F} = I_F - f(I_F, Θ) \bigodot (1 - M)$ (10)

    - 예측된 flare  $\widehat{F}$는 network input과 masked network output 사이의 차이로 계산됨

### Reconstruction algorithm - Post-processing for light source blending

- 저자들의 loss들은 명시적으로 saturated 영역 내의 어떤 “복원하기 위한 학습”을 network에서 막아준다.
  그래서, network의 출력은 임의적일 수 있음.
- 실제로, network는 광원을 제거하려 하기 때문에 다음 사진에서 보이듯이 주변 pixel들과 유사해진다.
- 목적은 flare를 제거하는 것이지 광원 자체를 제거하는 게 아님
  후 처리로 network 출력에 광원을 붙여준다.

![Untitled 6](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/fade4226-7f14-4560-a0b5-a005f438766d)

- key observation: 광원에 의한 flare는 입력 이미지에서 saturate될 가능성이 있음
  → 그러므로 강도에 기반해 쉽게 식별이 가능하다.
- 점진적인 변화를 주기 위해, **Reconstruction algorithm - Losses** 에서 정의된 마스크 M을 경계에서 페더링하여 $M_f$를 만든다.
  - 페더링: feature의 가장자리를 부드럽게 하거나 흐리게 만듬
- 선형공간에서 페더링된 마스크를 이용해서 입력과 출력 이미지들을 섞는다.
  $I_B = I_F \bigodot M_f + f(I_F, Θ) \bigodot (1 - M_f)$ (11)

---

## Experimental results

- semi-synthetic data로 훈련된 model들이 얼마나 일반화를 잘하는지 평가하기 위해 3가지 유형의 test_data 사용
  (a) ground truth를 가진 synthetic 이미지들
  (b) ground truth가 없는 실제 이미지들
  (c) ground truth가 있는 실제 이미지들
    - (c)를 얻기 위해서, 밖에 나가서 찍어옴
      하나는 artifact들을 만들기 위해 밝은 flare의 원인이 되는 광선이 들어올 수 있게 함
        다른 하나는 같은 광선을 막기 위해 illuminant와 카메라 사이의 시야에 occluder(가림막)를 둠

### Results - Comparison with prior work

- heuristics (발견법)
  - 불충분한 시간이나 정보로 인하여 합리적인 판단을 할 수 없거나, 체계적이면서 합리적인 판단이 굳이 필요하지 않은 상황에서 사람들이 빠르게 사용할 수 있게 보다 용이하게 구성된 간편추론의 방법
  - 문제해결에 있어서 복잡한 문제의 경우 초기에는 휴리스틱을 이용하여 과제를 단순화시킨 후 후기에 규범적(normative)인 의사결정 규칙을 사용하고, 단순한 task 상황에서는 처음부터 최종 의사결정에 이르기까지 규범적 규칙을 이용해 해결하려 한다는 가설
- 저자들은 논문에서 제안한 데이터 합성 기술을 범용적인 flare removal task에 적용할 수 있다고 설명
- dehazing, dereflection 알고리즘을 논문에서 제시한 데이터로 비교
  → haze와 reflection (흔한 flare artifacts)

![Untitled 7](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/90218d6e-1c0e-49af-a1bc-305166a8364e)

![Untitled 8](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/c69c7bcb-7f0e-46b2-9646-0c9508faaebe)

### Results - Ablation study

![Untitled 9](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/8a8d0499-a487-4fd2-a8b7-8acdcd1c828a)

- **flare loss**

  - 대부분의 flare들이 flare 뒤 장면보다 밝기 때문에, 저자들은 network가 모든 bright region들을 어둡게 학습하지 않도록 보장할 필요가 있음.

  ![Untitled 10](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/f551538a-c0d1-4f0a-87fb-ed82ec7b504a)

  - $L_F$가 없을 때, network는 flare의 일부분이 아니더라도 밝은 object의 몇몇 부분들을 제거하는 경향이 있음

- **Captured and simulated flare data**

  - captured data는 reflective flare

  - simulated data는 scattering flare

  - captured data, simulated data가 모두 필요하다는 것을 보여주기 위해, 저자들은 각각 하나의 source가 제외된 2개의 비교 모델들을 훈련 시킴

    - 예상한 대로 captured data, simulated data 모두 사용한 데이터들 보다 성능이 떨어진 것을 확인

    ![Untitled 11](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/21b8cd0a-64e8-4364-8b55-15d14be8ea8f)

### Results - Generalization

- **Across scenes**

  - 논문의 Semi-synthetic dataset은 다양한 패턴과 장면을 포함하므로, 훈련된 모델은 다양한 장면 유형에 걸쳐 일반화를 잘한다.
  - 모델은 대부분 시나리오에서 높은 퀄리티의 결과를 만들어냄

- **Across cameras**

  - 논문의 모든 reflective flare 훈련 이미지들은 focal length f=13mm 인 하나의 스마트폰 카메라로 찍음
  - 모델을 테스트할 때는 훈련을 제외한 다른 카메라 설계로 했음

  ![Untitled 12](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/bca94081-fa4f-4355-b9b0-0d1b8e376d7f)

  - 위 사진에서 보이듯이, 모델은 lens flare를 효과적으로 줄일 수 있다.
  - 모델이 일반화할 수 있는 범위에는 한계가 있음
    - 예를 들어, 모델은 fisheye와 같은 극도로 다른 lens로 찍힌 이미지들에서 잘 예측 못함.
    - 이는 특히 lens에 의존하는 반사 구성 요소에 해당
    - 후속 과제로 해결

### Results - High-resolution images

- 논문의 network는 $512 \times 512$ 이미지들로 훈련된다.

  - 논문의 방법을 더 높은 해상도 input에 적용하기 위한 본질적인 방법은 훈련 및 테스트를 할 때, 양쪽에서 16배 큰 대역폭을 요구되는 해상도 ($2048 \times 2048$)에서 훈련 시킨다.

  - 운이 좋게, lens flare가 주로 low-frequency artifact라는 사실을 활용할 수 있다.

    - 고해상도 이미지의 경우, 입력을 bilinear하게 다운 샘플링하고, 저해상도 flare-only 이미지를 예측하고, 다시 전체 해상도로 bilinear 하게 업 샘플링하고, 원래 입력에서 뺀다.

    ![Untitled 13](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/41e179a4-9b43-4c81-a820-837a6feff9d1)

  - 이는 고정된 저 해상도에서 훈련된 network를 상당한 품질 손실 없이 고해상도 이미지들로 처리할 수 있게 해준다.

![Untitled 14](https://github.com/sunjong5108/CAM-based_Flare_Removal_Network/assets/81843626/d2d3a32f-be07-40b4-8c8c-64538483444b)

<br/>

> **Reference** 
>
> [https://yichengwu.github.io/flare-removal/](https://yichengwu.github.io/flare-removal/)
