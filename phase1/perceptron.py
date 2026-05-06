import numpy as np

class Perceptron:
    def __init__(self, n_features, lr=0.1):
        # n_features 길이의 배열을 0으로 초기화
        self.weights = np.zeros(n_features)

        # self.bias를 0으로 초기화
        self.bias = 0

        # self.lr에 학습률 저장
        self.lr = lr

    def step_function(self, z):
        return np.where(z > 0, 1, 0)

    def forward(self, X):
        # 가중합 계산
        z = np.dot(X, self.weights) + self.bias

        # 예측값 반환
        return self.step_function(z)

    def fit(self, X, y, epochs=10):
        history = []

        for epoch in range(epochs):
            total_error = 0

            for i in range(len(X)):
                # 예측값 구하기
                y_pred = self.forward(X[i:i+1])[0]

                # 오차 계산
                # error = 정답 - 예측값
                error = y[i] - y_pred

                # 가중치 업데이트
                self.weights = self.weights + self.lr * error * X[i]

                # 편향 업데이트
                self.bias = self.bias + self.lr * error

                total_error += abs(error)

            # 에포크 끝: 전체 데이터에 대한 정확도 계산
            predictions = self.forward(X)
            accuracy = np.mean(predictions == y)
            history.append(accuracy)
            print(f"  Epoch {epoch+1:3d} | Accuracy: {accuracy:.2f} | Total Error: {total_error}")

            # 모든 데이터를 맞추면 조기 종료
            if accuracy == 1.0:
                print(f"{epoch+1} 에포크에서 수렴한다")
                break

        return history

    def predict(self, X):
        return self.forward(X)


# and 게이트 테스트
print("=" * 50)
print("AND 게이트 학습")
print("=" * 50)

# 학습 데이터
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# Perceptron 생성
p_and = Perceptron(n_features=2, lr=0.1)
history_and = p_and.fit(X_and, y_and, epochs=20)

# 결과 확인
print("\n결과:")
for x, y_true in zip(X_and, y_and):
    y_pred = p_and.predict(x.reshape(1, -1))[0]
    print(f"  {x} → {y_pred} (정답: {y_true})")

print(f"\n학습된 가중치: w = {p_and.weights}")
print(f"학습된 편향:   b = {p_and.bias}")


# or gate 테스트
print("\n" + "=" * 50)
print("OR 게이트 학습")
print("=" * 50)

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(n_features=2, lr=0.1)
history_or = p_or.fit(X_or, y_or, epochs=20)

print("\n결과:")
for x, y_true in zip(X_or, y_or):
    y_pred = p_or.predict(x.reshape(1, -1))[0]
    print(f"  {x} → {y_pred} (정답: {y_true})")