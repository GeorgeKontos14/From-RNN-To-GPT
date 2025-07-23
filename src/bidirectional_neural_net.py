import math

import torch

class BiNN:

    def __init__(self, d:int, p: int, k:int, q: int):
        self.d = d
        self.p = p
        self.k = k
        self.q = q

        self.W_x_f = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_h_f = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_y_f = torch.randn((p,q)) * 0.1

        self.H_f = []
        self.Z_f = []

        self.W_x_b = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_h_b = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_y_b = torch.randn((p,q)) * 0.1

        self.H_b = []
        self.Z_b = []

        self.dL_dWx_f = []
        self.dL_dWh_f = []
        self.dL_dWy_f = torch.zeros_like(self.W_y_f)

        self.dL_dWx_b = []
        self.dL_dWh_b = []
        self.dL_dWy_b = torch.zeros_like(self.W_y_b)


    def _forward_pass(self, X:torch.Tensor):
        T = X.shape[0]

        H_f = [torch.ones(T+1, self.p)*0.5 for _ in range(self.k)]
        Z_f = [torch.zeros(T, self.p) for _ in range(self.k)]

        H_b = [torch.ones(T+1, self.p)*0.5 for _ in range(self.k)]
        Z_b = [torch.zeros(T, self.p) for _ in range(self.k)]

        for t, x_t in enumerate(X):
            for l in range(self.k):
                if l == 0:
                    temp = x_t
                else:
                    temp = H_f[l-1][t+1]
                Z_f[l][t] = self.W_x_f[l]@temp + self.W_h_f[l]@H_f[l][t]
                H_f[l][t+1] = torch.tanh(Z_f[l][t])
        
        for t in range(T-1,-1,-1):
            for l in range(self.k):
                if l == 0:
                    temp = X[t]
                else:
                    temp = H_b[l-1][t]
                Z_b[l][t] = self.W_x_b[l]@temp+self.W_h_b[l]@H_b[l][t+1]
                H_b[l][t] = torch.tanh(Z_b[l][t])

        Y = H_f[self.k-1][1:]@self.W_y_f+H_b[self.k-1][:-1]@self.W_y_b

        self.H_f = torch.stack([h[1:] for h in H_f])
        self.Z_f = torch.stack(Z_f)

        self.H_b = torch.stack([h[:-1] for h in H_b])
        self.Z_b = torch.stack(Z_b)

        return Y
    
    def _backward_pass(self, X: torch.Tensor, Y: torch.tensor, Y_hat: torch.tensor):
        T = X.shape[0]

        dL_dh_f, dL_dz_f  = torch.zeros_like(self.H_f), torch.zeros_like(self.Z_f)
        dL_dh_b, dL_dz_b = torch.zeros_like(self.H_b), torch.zeros_like(self.Z_b)
        
        self.dL_dWx_f = [torch.zeros_like(w) for w in self.W_x_f]
        self.dL_dWh_f = [torch.zeros_like(w) for w in self.W_h_f]
        self.dL_dWx_b = [torch.zeros_like(w) for w in self.W_x_b]
        self.dL_dWh_b = [torch.zeros_like(w) for w in self.W_h_b]

        dL_dy = Y - Y_hat

        dL_dh_f[-1] = dL_dy@self.W_y_f.t()
        self.dL_dWy_f = self.H_f[-1].t()@dL_dy
    
        dL_dh_b[-1] = dL_dy@self.W_y_b.t()
        self.dL_dWy_b = self.H_b[-1].t()@dL_dy

        for t in range(T):
            for l in range(self.k-1, -1, -1):
                dL_dz_f[l][t] = dL_dh_f[l][t] * (1-torch.square(self.H_f[l][t]))
                
                dL_dz_b[l][t] = dL_dh_b[l][t] * (1-torch.square(self.H_b[l][t]))

                if t > 0:
                    h_prev = self.H_f[l][t-1]
                else:
                    h_prev = torch.ones_like(self.H_f[l][0])*0.5

                if t < T-1:
                    h_next = self.H_b[l][t+1]
                else:
                    h_next = torch.ones_like(self.H_b[l][-1])*0.5

                self.dL_dWh_f[l] += dL_dz_f[l][t].view(-1, 1) @ h_prev.view(1, -1)
                self.dL_dWh_b[l] += dL_dz_b[l][t].view(-1, 1) @ h_next.view(1, -1)

                if l >= 1:
                    dL_dh_f[l-1][t] = dL_dz_f[l][t]@self.W_x_f[l].t()
                    dL_dh_b[l-1][t] = dL_dz_b[l][t]@self.W_x_b[l].t()
        
        self.dL_dWx_f[0] = dL_dz_f[0].t() @ X
        self.dL_dWx_b[0] = dL_dz_b[0].t() @ X
        for l in range(1, self.k):
            self.dL_dWx_f[l] = dL_dz_f[l].t()@self.H_f[l-1]
            self.dL_dWx_b[l] = dL_dz_b[l].t()@self.H_b[l-1]


    def fit(
            self, 
            X:torch.Tensor, 
            Y_hat: torch.Tensor, 
            alpha:float = 0.1, 
            epochs:int = 10, 
            verbose:bool = False, 
            decay:str = "const", 
            decay_rate:float = 0.9
        ):
        for ep in range(epochs):
            if decay == "exp":
                a = alpha*math.exp(-decay_rate*ep/epochs)
            elif decay == "inv":
                a = alpha/(1+decay_rate*ep/epochs)
            else:
                a = alpha

            Y = self._forward_pass(X)
            self._backward_pass(X, Y, Y_hat)

            for l in range(self.k):
                self.W_x_f[l] -= a*self.dL_dWx_f[l]
                self.W_h_f[l] -= a*self.dL_dWh_f[l]
                self.W_x_b[l] -= a*self.dL_dWx_b[l]
                self.W_h_b[l] -= a*self.dL_dWh_b[l]
            
            self.W_y_f -= a*self.dL_dWy_f
            self.W_y_b -= a*self.dL_dWy_b

            loss = torch.mean((Y - Y_hat) ** 2).item()
            if verbose:
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.6f}")