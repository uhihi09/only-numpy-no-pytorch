import matplotlib.pyplot as plt
from perceptron import history_and
from perceptron import history_or
from xor_perceptron import history_xor

fig = plt.figure()

plt.plot(history_and, label='AND')
plt.plot(history_or, label='OR')
plt.plot(history_xor, label='XOR')

plt.title('accuracy history')

plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.legend(loc='best')

plt.show()