import numpy as np
from perceptron import Perceptron

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

print("=" * 50)
print("XOR 게이트 학습")
print("=" * 50)

p_xor = Perceptron(n_features=2, lr=0.1)
history_xor = p_xor.fit(X_xor, y_xor, epochs=20)

print("\n결과:")
for x, y_true in zip(X_xor, y_xor):
    y_pred = p_xor.predict(x.reshape(1,-1)[0])
    print(f"  {x} → {y_pred} (정답: {y_true})")