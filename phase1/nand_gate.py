import numpy as np

def manual_nand_gate(x1, x2):
    w1 = -0.5
    w2 = -0.5
    b  = 1

    z = w1*x1 + w2*x2 + b
    return np.where(z > 0,1,0)

# 테스트
print("=== 수동 NAND 게이트 ===")
for x1, x2 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
    result = manual_nand_gate(x1, x2)
    print(f"  {x1} NAND {x2} = {result}")