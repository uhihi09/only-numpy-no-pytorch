import numpy as np

def manual_and_gate(x1, x2):
    # w1, w2, b 값 정하기
    w1 = 1
    w2 = 1
    b  = -1

    # 가중합 z 계산
    z = w1*x1 + w2*x2 + b

    # z > 0이면 1, 아니면 0을 리턴
    return np.where(z > 0, 1, 0)

# 테스트
print("=== 수동 AND 게이트 ===")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = manual_and_gate(x1, x2)
    print(f"  {x1} AND {x2} = {result}")