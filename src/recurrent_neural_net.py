import torch

class RNN:

    def __init__(self, d:int, p: int, k:int, q: int):
        self.d = d
        self.p = p
        self.k = k
        self.q = q

        self.W_x = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_h = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_y = torch.randn((p,q)) * 0.1

        self.H = []
        self.Z = []

        self.dL_dWx = []
        self.dL_dWh = []
        self.dL_dWy = torch.zeros_like(self.W_y)

    def _forward_pass(self, X: torch.Tensor):
        T = X.shape[0]

        Y = torch.zeros(T, self.q)
        H = [torch.zeros(T+1, self.p) for _ in range(self.k)]
        Z = [torch.zeros(T, self.p) for _ in range(self.k)]

        for t, x_t in enumerate(X):
            for l in range(self.k):
                if l == 0:
                    temp = x_t
                else:
                    temp = H[l-1][t+1]
                Z[l][t] = self.W_x[l]@temp + self.W_h[l]@H[l][t]
                H[l][t+1] = torch.tanh(Z[l][t])
            Y[t] = self.W_y.t()@H[self.k-1][t+1]

        self.H = torch.stack([h[1:] for h in H])
        self.Z = torch.stack(Z)

        return Y
        

    def _backward_pass(self, X: torch.Tensor, Y: torch.tensor, Y_hat: torch.tensor):
        T = X.shape[0]

        dL_dh, dL_dz  = torch.zeros_like(self.H), torch.zeros_like(self.Z)
        
        self.dL_dWx = [torch.zeros_like(w) for w in self.W_x]
        self.dL_dWh = [torch.zeros_like(w) for w in self.W_h]

        dL_dy = Y - Y_hat

        dL_dh[-1] = dL_dy@self.W_y.t()
        self.dL_dWy = self.H[-1].t()@dL_dy

        for t in range(T):
            for l in range(self.k-1, -1, -1):
                dL_dz[l][t] = dL_dh[l][t] * (1-torch.square(self.H[l][t]))

                if t > 0:
                    h_prev = self.H[l][t-1]
                else:
                    h_prev = torch.zeros_like(self.H[l][0])
                self.dL_dWh[l] += dL_dz[l][t].view(-1, 1) @ h_prev.view(1, -1)

                if l >= 1:
                    dL_dh[l-1][t] = dL_dz[l][t]@self.W_x[l].t()
        
        self.dL_dWx[0] = dL_dz[0].t() @ X
        for l in range(1, self.k):
            self.dL_dWx[l] = dL_dz[l].t()@self.H[l-1]
        
    def fit(self, X:torch.Tensor, Y_hat: torch.Tensor, alpha:float = 0.1, epochs:int = 10, verbose:bool = False):
        for ep in range(epochs):
            Y = self._forward_pass(X)
            self._backward_pass(X, Y, Y_hat)

            for l in range(self.k):
                self.W_x[l] -= alpha*self.dL_dWx[l]
                self.W_h[l] -= alpha*self.dL_dWh[l]
            
            self.W_y -= alpha*self.dL_dWy

            loss = torch.mean((Y - Y_hat) ** 2).item()
            if verbose:
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.6f}")
                