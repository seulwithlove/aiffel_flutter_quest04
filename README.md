# [Flutter-Quest04] Repository

<Flutter - 5. Flutter Networking> 를 학습한 후 진행한 메인퀘스트 레포지토리

## 파일 구성

1. quest04
   - 책을 학습후 문제대로 앱 구현
   - 구현 결과 영상, 사진
     - 웹 배포 [🔗링크](https://seulwithlove.github.io/aiffel_flutter_quest04/)
     - ![Screenshot 2024-02-15 at 22 35 06](https://github.com/seulwithlove/aiffel_flutter_quest04/assets/140625136/8ece221c-5966-46fa-af1a-caa115a8cc0e)
     - ![Screenshot 2024-02-15 at 22 38 13](https://github.com/seulwithlove/aiffel_flutter_quest04/assets/140625136/6b247825-afec-470b-afdb-ceff44e9d60f)
    
     - https://github.com/seulwithlove/aiffel_flutter_quest04/assets/140625136/95217e6b-45b0-48fc-b95c-c6b885ba74ef


## 회고
**[Keep]**
- 웹배포 방식에 대한 이해가 완료되지 않은 상황이라 포기하고 싶었지만 포기하지 않고 끝까지 결과물을 만들어냄!

**[Problem]**
1. py 파일을 만들어서 python 파일을 터미널에서 실행시키고, ngrok로 실시간 API 송신 결과를 확인 하는 작업을 하기 위한 전반에 대한 이해가 부족했음
2. 깃허브에 웹배포를 하는 방식에 대해 받은 자료가 깃허브 업데이트 내용을 담고 있지 않아서 계속 알수 없는 에러가 생김
3. W&B에서 실험했던 모델은 자동으로 파라미터를 조정해줬던 상황이라 어떻게 다운받아야할지 모름

**[Try]**
- 1번: 양희님과 이런저런 시도를 해보고, 그루 선재님의 코드를 베이스 삼아서 py파일의 코드 구조를 파악하고 우리 모델을 가져오는 부분에 맞춰 flutter 코드를 수정(선재님🙏)
- 2번: 그루 강훈님의 도움을 받아서 README.md 파일이 무조건 레포지토리에 있어야 한다는 사실을 알게되었고, 내 깃헙계정이름으로 github.io 주소의 깃헙 블로그부터 생성해서 문제를 해결(강훈님🙏)
- 3번: 그루 강훈님이 W&B에서 자동으로 저장한 최적의 모델을 다운받는 방법을 알려주셔서 바로 해결(강훈님🙏🙏)

**종합**
여러모로 포기하고 싶은 마음이 정말 컸던 퀘스트였지만, 밤늦게까지 남아 함께 작업을 하고 결과까지 만들어낼수 있어서 매우 뿌듯함. 하지만 웹 배포, 상태값을 가져오는것, Navigator 등의 부분에 대한 공부가 필요하다고 다시 한번 느낌.



---

## AIFFEL Campus Online Code Peer Review

- Coder : 김양희, 이슬
- Reviewr : 


# PRT(Peer Review Template)
- [ ]  **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 문제에서 요구하는 최종 결과물이 첨부되었는지 확인
    - 문제를 해결하는 완성된 코드란 프로젝트 루브릭 3개 중 2개, 
    퀘스트 문제 요구조건 등을 지칭
        - 해당 조건을 만족하는 코드를 캡쳐해 근거로 첨부
윅부분에 잘 기록이 되어있다. 
    
- [ ]  **2. 전체 코드에서 가장 핵심적이거나 가장 복잡하고 이해하기 어려운 부분에 작성된 
주석 또는 doc string을 보고 해당 코드가 잘 이해되었나요?**
    - 해당 코드 블럭에 doc string/annotation이 달려 있는지 확인
    - 해당 코드가 무슨 기능을 하는지, 왜 그렇게 짜여진건지, 작동 메커니즘이 뭔지 기술.
    - 주석을 보고 코드 이해가 잘 되었는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


        
- [ ]  **3. 에러가 난 부분을 디버깅하여 문제를 “해결한 기록을 남겼거나” 
”새로운 시도 또는 추가 실험을 수행”해봤나요?**
    - 문제 원인 및 해결 과정을 잘 기록하였는지 확인
    - 문제에서 요구하는 조건에 더해 추가적으로 수행한 나만의 시도, 
    실험이 기록되어 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.



- [ ]  **4. 회고를 잘 작성했나요?**
    - 주어진 문제를 해결하는 완성된 코드 내지 프로젝트 결과물에 대해
    배운점과 아쉬운점, 느낀점 등이 기록되어 있는지 확인
    - 전체 코드 실행 플로우를 그래프로 그려서 이해를 돕고 있는지 확인
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.


        
- [ ]  **5. 코드가 간결하고 효율적인가요?**
    - 파이썬 스타일 가이드 (PEP8) 를 준수하였는지 확인
    - 하드코딩을 하지않고 함수화, 모듈화가 가능한 부분은 함수를 만들거나 클래스로 짰는지
    - 코드 중복을 최소화하고 범용적으로 사용할 수 있도록 함수화했는지
        - 잘 작성되었다고 생각되는 부분을 캡쳐해 근거로 첨부합니다.





# 참고 링크 및 코드 개선
```
# 코드 리뷰 시 참고한 링크가 있다면 링크와 간략한 설명을 첨부합니다.
# 코드 리뷰를 통해 개선한 코드가 있다면 코드와 간략한 설명을 첨부합니다.
```
