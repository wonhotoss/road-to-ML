The spelled-out intro to neural networks and backpropagation: building micrograd 시청중

미분과 기울기의 의미

Value 클래스에 대해 설명중. 많이들 쓰는 lazy evaluation용 값 wrapper

value as a graph node. value는 자식 노드들간의 연산 결과

그릴 때는 graphviz.Digraph

expression node + graph + draw... 아직은 뻔한 내용들.

이제 backpropargation. node가 grad를 갖게 된다. 이걸 어떻게 찾는지.

inline gradient

아래로 내려가며 수동으로 derivation 찾기. 반복.

chain rule. product of all chain along down from the top

grad: 결과 노드에서 이 노드까지의 back propargation 결과(not local gradient)

chain rule 을 반복해서 끝에서 처음까지 grad를 찾아오며 훑는다.

절차적으로 grad를 쓰면서 내려오고 난 후, 산술방식으로(입력에 작은 값을 더한 후 결과 변화를 관찰) grad 검증.

manual back propargation

tanh operation

long and exhausting manual back propargation => automatic?

_backward. 자식들(입력)의 grad를 지정한다. 부모 -> 자식방향 recurdion. _backward 를 재귀호출하지 않아도 되나?

expression은 꼭 tree가 아닐 수도 있으니..topological sort 필요

a + a = result 에서 a 의 grad가 1로 나온다. 두 항이 같은 객체라서 보기에만 틀려 보이는 거 아닌가

더 복잡한 식으로 가면, 하나의 값이 복수의 입력으로 쓰일 때 grad가 어느 하나의 채널에서 온 값만 채택해서 생기는 문제.

accumulate gradients

_backward 전에 초기화해야 하지 않나

rmul, radd

truediv?

__pow__ 의 backward는 직접 구현

만들어볼 것. 
value graph 생성기. 
+ - * / pow atan exp
back propargation. 임의의 값과 expression으로 산술적으로(delta)스스로를 검증하는 테스트수트. 

왜 pow는 rhand에 상수만 받는가?

그래프 좀 짧게

1. 임의의 value graph 생성
2. 임의의 neural net 생성. 1과 같은 수의 입력. 임의의 layers * neurons
3. 1의 결과와 비교하느 loss function
4. gradient 따라가며 최적화(loss 최소화)

- node 껍데기. 간단한 operation만
- draw
- backward. topological sort + recursion
- 임의 expression 생성기
- 산술적 테스트
- expression -> torch 변환식
- torch 비교 테스트

