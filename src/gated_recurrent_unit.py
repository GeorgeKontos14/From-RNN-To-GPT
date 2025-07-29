import math

import torch

class GRU:

    def __init__(self, d:int, p: int, k:int, q: int):
        self.d = d
        self.p = p
        self.k = k
        self.q = q

        self.W_x_z = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_x_r = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        
        self.V_x = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]

        self.W_h_z = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_h_r = [torch.randn((p,p)) * 0.1 for _ in range(k)]

        self.V_h = [torch.randn((p,p)) * 0.1 for _ in range(k)]

        self.W_y = torch.randn((p,q)) * 0.1

        self.H = []

        self.Z = []
        self.R = []
        self.H_ = []

        self.dL_dWx_z = []
        self.dL_dWx_r = []

        self.dL_dVx = []

        self.dL_dWh_z = []
        self.dL_dWh_r = []

        self.dL_dVh = []

    def _forward_pass(self, X: torch.Tensor):
        T = X.shape[0]

        Y = torch.zeros(T, self.q)
        H = [torch.zeros(T+1, self.p) for _ in range(self.k)]
        Z = [torch.zeros(T, self.p) for _ in range(self.k)]
        R = [torch.zeros(T, self.p) for _ in range(self.k)]
        H_ = [torch.zeros(T, self.p) for _ in range(self.k)]

        for t, x_t in enumerate(X):
            for l in range(self.k):
                temp = x_t if l == 0 else H[l-1][t+1]

                Z[l][t] = torch.sigmoid(self.W_x_z[l]@temp+self.W_h_z[l]@H[l][t])
                R[l][t] = torch.sigmoid(self.W_x_r[l]@temp+self.W_h_r[l]@H[l][t])
                H_[l][t] = torch.tanh(self.V_x[l]@temp+self.V_h[l]@(H[l][t]*R[l][t]))
                H[l][t+1] = Z[l][t]*H[l][t]+(1-Z[l][t])*H_[l][t]

            Y[t] = self.W_y.t()@H[self.k-1][t+1]

        self.H = torch.stack([h[1:] for h in H])
        self.Z, self.R, self.H_ = torch.stack(Z), torch.stack(R), torch.stack(H_)

        return Y

    def _backward_pass(self, X: torch.Tensor, Y: torch.tensor, Y_hat: torch.tensor):
        T = X.shape[0]

        dL_dh, dL_dz, dL_dr, dL_dh_ = torch.zeros_like(self.H), torch.zeros_like(self.Z), torch.zeros_like(self.R), torch.zeros_like(self.H_)

        self.dL_dWx_z = [torch.zeros_like(w) for w in self.W_x_z]
        self.dL_dWx_r = [torch.zeros_like(w) for w in self.W_x_r]

        self.dL_dVx = [torch.zeros_like(w) for w in self.V_x]

        self.dL_dWh_z = [torch.zeros_like(w) for w in self.W_h_z]
        self.dL_dWh_r = [torch.zeros_like(w) for w in self.W_h_r]

        self.dL_dVh = [torch.zeros_like(w) for w in self.V_h]

        dL_dy = Y - Y_hat

        dL_dh[-1] = dL_dy@self.W_y.t()
        self.dL_dWy = self.H[-1].t()@dL_dy

        for t in range(T):
            for l in range(self.k-1, -1, -1):
                dL_dh_[l][t] = dL_dh[l][t]*(1-self.Z[l][t])
                e_h = dL_dh_[l][t]*(1-self.H_[l][t]**2)

                if t > 0:
                    h_prev = self.H[l][t-1].view(1,-1)
                else:
                    h_prev = torch.zeros_like(self.H[l][0]).view(1,-1)

                dL_dz[l][t] = dL_dh[l][t]*(h_prev-self.H_[l][t])
                e_z = dL_dz[l][t]*self.Z[l][t]*(1-self.Z[l][t])
                dL_dr[l][t] = e_h@self.V_h[l]@h_prev.t()
                e_r = dL_dr[l][t]*self.R[l][t]*(1-self.R[l][t])

                self.dL_dWh_z[l] += e_z.view(-1,1)@h_prev
                self.dL_dWh_r[l] += e_r.view(-1,1)@h_prev
                self.dL_dVh[l] += e_h.view(-1,1)@(h_prev*self.R[l][t].view(1,-1))

                if l >= 1:
                    dL_dh[l-1][t] = e_z@self.W_x_z[l].t()+e_r@self.W_x_r[l].t()+e_h@self.V_x[l].t()
                
                    h_prev_l = self.H[l-1][t].view(1,-1)
                else:
                    h_prev_l = X[t].view(1,-1)

                self.dL_dWx_z[l] += e_z.view(-1,1)@h_prev_l
                self.dL_dWx_r[l] += e_r.view(-1,1)@h_prev_l
                self.dL_dVx[l] += e_h.view(-1,1)@h_prev_l

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
                self.W_x_z[l] -= a*self.dL_dWx_z[l]
                self.W_x_r[l] -= a*self.dL_dWx_r[l]
                self.V_x[l] -= a*self.dL_dVx[l]

                self.W_h_z[l] -= a*self.dL_dWh_z[l]
                self.W_h_r[l] -= a*self.dL_dWh_r[l]
                self.V_h[l] -= a*self.dL_dVh[l]
            
            self.W_y -= a*self.dL_dWy

            loss = torch.mean((Y - Y_hat) ** 2).item()
            if verbose:
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.6f}")