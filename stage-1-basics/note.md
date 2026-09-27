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
- 산술적 테스트
- 임의 expression 생성기
- 임의 expression 생성기 + 산술적 테스트
- pow
- exp
- atan
- 임의 expression 생성기(atan + exp 포함) + 산술적 테스트
- torch test
- 임의 expression 생성기 tree => graph
- expression: embed torch node
- 임의 expression 생성기 + torch 테스트

- neural network(neuron - layer - mlp)
- 고정 입력과 임의의 expression. neural network으로 추적

### 리뷰 이후

- topological sort
- 전위 DFS 와 topological sort 가 만드는 차이를 설명

"a + a = result 에서 a 의 grad가 1로 나온다. 두 항이 같은 객체라서 보기에만 틀려 보이는 거 아닌가

더 복잡한 식으로 가면, 하나의 값이 복수의 입력으로 쓰일 때 grad가 어느 하나의 채널에서 온 값만 채택해서 생기는 문제."

위의 노트에서 발췌

위상 정렬이 틀어지면서, backward propargation의 어느 한 경로가 차단된 결과를 낳는다. 끊어진 시냅스? 하지만 foward 경로는 살아있다. foward 경로까지 끊어지면 다른 경로가 유실된 backward 값을 물려받을 듯.


a = b + c
b = c * 2
c = ...
=> 위상 정렬이 틀어지면, c가 B에서 오는 back-propargation을 받지 못하게 된다.
=> 그냥 recursive하게 
    a => b => c => ...
    a => c => ...
이러면 c(와 자식들)가 a로부터 두 번 전파받는 식이 된다. grad(a) + (grad(a) * 2)... 그런데 이러면 맞지 않나? 확인..

a = Expression(1)
r = a + a
test_grad(r)
-> 여기선 재귀도 괜찮다

a = Expression(1)
b = a * Expression(1)
r = b + b
test_grad(r)

original [4.0, 4.0]
estimated [2.0000000000131024, 2.0000000000131024]
torch [2.0, 2.0]

-> 여기선 문제가 생긴다.
r = (a * 1) + (a * 1) = 2 * a 이니 a의 기울기는 2가 맞다.
backward 경로에선 1만 유통되니, 기울기가 4라면 a 가 4번 방문되었다는 건데 어떻게?

backward() 자체가 형제들에게 한 번에 기울기를 나눠주는 식이라 DFS가 제대로 기능하지 않았다. 자식 형제들한테 한 번에 기울기를 나눠주는데, 두 자식이 사실 같은 놈이라 한 놈한테 두 번 나눠주었다. 여기까진 맞는데, 재귀적으로 호출되면서 (큰아들 -> 공통손자 + 작은아들 -> 공통손자)여야 하는데((큰아들 + 작은아들) -> 공통손자) * 2 가 되어 버렸다. 이건 애초에 구현이 backward()가 breadth first 호출을 염두에 두고 있기 때문에(self.grad += ...). recursive_backward 를 따로 만들자.

일이 커질 것 같다. recursive하게 구현하려면, node 들이 매 backward call마다 위에서 내려온 grad만 밑으로 내려보내야 하고, 최종 결과만 합산해서 반환해야 한다.기존의 self.grad += ... 하는 식으론 안된다. 한 번 구현해보자. 

dfs_backward 잘 기능한다. 객체 안에 쌓는 것보다 이게 더 낫다. 반복 테스트. grad 값은 맞는 것 같은데 math domain error가 너무 많이 난다

math domainerror, dividebyzero...이건 일단 무시

음수를 실수 지수로 거듭제곱할 때. 지수는 정수로.

이걸 MLP로. 동작한다

torch 미러가 갱신을 따라가지 않는다. p.value += … 뒤에도 p.tensor는 초기값이다. 학습 도중·후에 torch 대조를 하면 가짜 불일치가 난다.
torch 쪽 .grad는 누적된다. 

forward가 노드 생성 때마다 그래프 전체를 재평가한다. - 

앞 두 층 기울기가 틀린데 왜 loss가 0으로 갔는가? 출력층 파라미터 30개와 데이터 점 3개를 놓고 생각하라.

dataset과 expects는 서로 독립인 난수다. 입력과 정답 사이에 배울 함수가 없다. 그런데 완벽히 맞췄다. 이 네트워크가 배운 것은 무엇인가? 학습에 안 쓴 입력을 넣으면 어떤 값이 나오는가?

[ ] 위상정렬 수정 후 재사용 식·create_random_graph(5)·MLP 세 층 전부 torch 일치, assert로
[ ] 배울 함수가 있는 데이터로 다시 학습, loss 곡선, 학습에 안 쓴 입력에서의 결과
[ ] note.md "막혔던 지점 목록" — 비어 있다. 이번 위상정렬 건이 첫 항목이다

막혔던 지점 목록

backward()를 재귀적으로 전환하는 시점에서 기존 동영상에서 본 구현 - 노드에 기울기를 쌓는 - 방식과 충돌해서 헤멤. 지금 방식에선 위상 정렬이 필요하지 않다. 필요한 횟수만큼 방문하고 방문 당시의 gradient가 전파되고 최종 결과만 총 방문에서 합산. gradient 클리어도 필요없다.

생각해보면 노드에 grad를 쌓고 클리어하는 건 강의의 구현에선 피할 수 없는 부분인데(복수 output이 input에 grad를 다 쌓을 때까지 전파가 지연되므로), 정작 torch는 retain_grad() opt-in인 걸 보면 중간 노드에는 grad를 쌓지 않는 depth first 재귀 방식이 아닐까?

여기서 리뷰 후 재구현

카운트가 0이 되는 순간을 이미 알고 있다. 지금 backward()는 매 스텝 refcounts를 다시 훑어 0을 찾는다. 감소시키는 자리에서 0이 된 노드를 준비 목록에 넣으면 스캔이 사라진다. 다시 쓰는 김에 볼 것.
**의 더미 상수 피연산자. 규칙 표가 지수를 o1.value에서 읽으려고 만든 가짜 노드다. 노드가 부가 데이터를 들 수 있다면 필요한가. test_grad가 잎을 고를 때 op가 비어 있지 않다는 이유로 걸러내는 것도 그 더미 때문이다.
dict의 키는 객체 동일성이다. Expression에 __eq__를 정의하는 순간 해시가 깨진다. 지금은 문제없고, 앞으로도 값 비교를 붙이지 말라는 뜻이다.
그래프에 안 닿은 파라미터. 학습 루프가 gradients[p]로 읽으니 도달하지 못한 파라미터는 KeyError다. 기울기 0이 정답인가, 크래시가 정답인가. 당신 원칙대로면 후자일 텐데, 의도한 것인지만 확인하라.