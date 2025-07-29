import torch
import numpy as np
import matplotlib.pyplot as plt
from long_short_term_memory import LSTM
from recurrent_neural_net import RNN
from bidirectional_neural_net import BiNN
from gated_recurrent_unit import GRU

# --- 1. Generate SINGLE (T,d) and (T,q) sequence ---
def generate_sequence(T=50, d=3, q=1):
    """Generate ONE (T,d) input and (T,q) output sequence"""
    t = np.linspace(0, 10, T)
    X = np.zeros((T, d))
    Y = np.zeros((T, q))
    
    # Create d input features
    for feature in range(d):
        freq = np.random.uniform(0.1, 0.5)
        phase = np.random.uniform(0, 2*np.pi)
        X[:, feature] = np.sin(2*np.pi*freq*t + phase) + 0.1*np.random.randn(T)
    
    # Create output (simple linear combination)
    Y[:, 0] = 0.5*X[:, 0] + 0.3*X[:, 1] - 0.2*X[:, 2]
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# Generate data
X_train, Y_train = generate_sequence(T=50, d=3, q=1)  # Single sequence
X_test, Y_test = generate_sequence(T=50, d=3, q=1)     # Another single sequence

print("Shapes:")
print(f"X_train: {X_train.shape} (should be (T,d))")
print(f"Y_train: {Y_train.shape} (should be (T,q))")

# --- 2. Initialize LSTM ---
neural_net = GRU(d=3, p=32, k=2, q=1)  # Matches input/output dims

# --- 3. Train on SINGLE sequence ---
neural_net.fit(X_train, Y_train, 
         alpha=0.1, epochs=250, verbose=True, decay="exp")

# --- 4. Predictions ---
with torch.no_grad():
    Y_train_pred = neural_net._forward_pass(X_train)  # Direct (T,q) output
    Y_test_pred = neural_net._forward_pass(X_test)

# --- 5. Plotting ---
plt.figure(figsize=(15, 5))

# Training results
plt.subplot(1, 2, 1)
plt.plot(Y_train[:, 0], label='True')
plt.plot(Y_train_pred[:, 0], '--', label='Predicted')
plt.title(f'Training Sequence\n(X: {X_train.shape}, Y: {Y_train.shape})')
plt.xlabel('Timesteps')
plt.legend()

# Test results
plt.subplot(1, 2, 2)
plt.plot(Y_test[:, 0], label='True')
plt.plot(Y_test_pred[:, 0], '--', label='Predicted')
plt.title(f'Test Sequence\n(X: {X_test.shape}, Y: {Y_test.shape})')
plt.xlabel('Timesteps')
plt.legend()

plt.tight_layout()
plt.show()

# --- Evaluation ---
train_mse = torch.mean((Y_train_pred - Y_train)**2).item()
test_mse = torch.mean((Y_test_pred - Y_test)**2).item()
print(f'\nMSE:\nTrain: {train_mse:.5f}\nTest: {test_mse:.5f}')