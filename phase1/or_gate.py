import numpy as np

def manual_or_gate(x1, x2):
    w1 = 1
    w2 = 1
    b = -0.5

    z = w1*x1 + w2*x2 + b

    return np.where(z > 0, 1, 0)

# 테스트
print("=== 수동 OR 게이트 ===")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = manual_or_gate(x1, x2)
    print(f"  {x1} OR {x2} = {result}")