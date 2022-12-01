# Block Arrangement Environment (BAE)
BAE는 조선소 내 블록 배치 정책을 학습할 수 있는 강화학습 환경입니다.

## 프로젝트 배경
조선소에서 건조되는 선박은 다수의 블록으로 나누어 만들어집니다. 블록은 다양한 이유로 블록 적치장에 임시 적치되어 작업을 대기하는 일이 빈번히 발생합니다. 블록 적치장 운용 시, 블록 적치 작업 및 블록 반출 작업의 시각을 미리 알 수 없는 경우가 흔해 미리 적치장 운용 계획을 수립이 어렵습니다. 

블록 운반 작업은 시간 및 금전적 비용 소모가 크므로 불필요한 블록 운반 작업의 최소화가 중요하며, 블록 재배치 작업은 대표적인 불필요한 블록 운반 작업입니다. 블록 재배치 작업은 아래 그림과 같이 적치장 안쪽의 블록에 접근 및 반출이 어려울 때, 다른 블록들을 적치장 내 다른 위치로 이동시키는 작업입니다. 재배치 작업의 최소화는 효율적인 적치장 운용을 위해 가장 중요한 요소이며, **BAE는 재배치 작업을 최소화하는 블록 배치 정책을 학습할 수 있는 강화학습 환경입니다.**

![intro](https://user-images.githubusercontent.com/31296656/204964974-2b2de575-a63e-44b0-8e6d-2050d265c203.PNG)
<br>

## 프로젝트 소개
![figure5](https://user-images.githubusercontent.com/31296656/204965211-61f20ce7-82d0-4eee-b54d-8514f04ace1e.PNG)
<br>

BAE는 적치장 내에서 발생하는 블록 운반 작업 시뮬레이션 기반 강화학습 환경입니다. BAE의 특징은 블록 적치장의 크기 및 블록 운반 작업 스케줄의 특성을 고려하여 블록 배치 정책을 수립할 수 있으며, 적치 작업뿐만 아니라 재배치 작업 시에도 미래에 발생할 재배치 작업을 최소화하도록 설계되었습니다. 또한 블록 적치 기간 정보를 미리 알 수도 있고 모를 수도 있는 현실의 상황을 반영하여 에이전트에게 제공되는 블록 적치 기간 정보를 조절하여 학습할 수 있습니다. 위 그림은 BAE와 에이전트가 상호작용하며 학습하는 프로세스를 나타낸 것이며 세부 내용은 논문을 참조하시기 바랍니다.

## 프로젝트 이용 방법
BAE는 오픈소스 강화학습 프레임워크 [JORLDY](https://github.com/kakaoenterprise/JORLDY)에 기반하여 제작되었으며, 세부 환경 설정 및 조작 방법은 [이 논문](https://arxiv.org/abs/2204.04892)을 참조하시기 바랍니다.

jorldy기반 학습 방법은 아래와 같습니다.
1. jorldy 설치 및 BAE 다운로드
2. /JORLDY/jorldy/config 내 각 알고리즘 폴더에 BAE/config에 있는 파일 추가
3. /JORLDY/jorldy/env에 Block.py, Space.py, blockArrangement.py 파일 추가
4. /JORLDY/jorldy/env 내 _env_dict.txt에 ('block_arrangement', <class 'core.env.BlockArrangement.block_arrangement'>) 추가
5. python main.py --config config.dqn.block_arrangement 

## 프로젝트 결과
아래 그림은 PPO 알고리즘 기반 
![PPO_example](https://user-images.githubusercontent.com/31296656/204970196-b0334919-ccb0-4a24-85be-884bfa8c7c30.png)

우리는 BAE에 다양한 아이디어가 적용되어 더욱 실용적이고 뛰어난 성능의 블록 배치 정책이 제안되길 희망하여 다양한 시나리오에 대한 벤치마크 성능을 제공합니다. 아래 표는 본 프로젝트에서 수행한 실험 시나리오이며, 자세한 실험 결과 및 분석 내용은 논문을 참조하시기 바랍니다.
![시나리오](https://user-images.githubusercontent.com/31296656/204969318-cdf9d35a-7161-4120-b669-632653ce1326.PNG)

