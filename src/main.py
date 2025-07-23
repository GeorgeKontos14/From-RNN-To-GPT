import torch
import matplotlib.pyplot as plt
import numpy as np
from recurrent_neural_net import RNN

T = 100
d = 1  
q = 1    

x_vals = np.linspace(0, 4 * np.pi, T + 1)
sine_wave = np.sin(x_vals)

X = torch.tensor(sine_wave[:-1], dtype=torch.float32).reshape(T, d)

Y_hat = torch.tensor(sine_wave[1:], dtype=torch.float32).reshape(T, q)

p = 10   # hidden size
k = 2    # number of layers

rnn = RNN(d=d, p=p, k=k, q=q)

rnn.fit(X, Y_hat, alpha=0.01, epochs=200, verbose=True)

with torch.no_grad():
    Y_pred = rnn._forward_pass(X)

plt.plot(Y_hat.numpy(), label='True')
plt.plot(Y_pred.numpy(), label='Predicted')
plt.legend()
plt.title("Sine Wave Prediction")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
