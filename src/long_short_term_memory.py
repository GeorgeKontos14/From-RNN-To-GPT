import math

import torch

class LSTM:

    def __init__(self, d:int, p: int, k:int, q: int):
        self.d = d
        self.p = p
        self.k = k
        self.q = q

        self.W_x_i = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_x_f = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_x_o = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]
        self.W_x_c = [torch.randn((p,p)) * 0.1 if l > 0 else torch.randn((p,d)) * 0.1 for l in range(k)]


        self.W_h_i = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_h_f = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_h_o = [torch.randn((p,p)) * 0.1 for _ in range(k)]
        self.W_h_c = [torch.randn((p,p)) * 0.1 for _ in range(k)]

        self.W_y = torch.randn((p,q)) * 0.1

        self.H = []
        self.C = []

        self.I = []
        self.F = []
        self.O = []
        self.C_n = []

        self.dL_dWx_i = []
        self.dL_dWx_f = []
        self.dL_dWx_o = []
        self.dL_dWx_c = []
        
        self.dL_dWh_i = []
        self.dL_dWh_f = []
        self.dL_dWh_o = []
        self.dL_dWh_c = []

        self.dL_dWy = torch.zeros_like(self.W_y)

    def _forward_pass(self, X: torch.Tensor):
        T = X.shape[0]

        Y = torch.zeros(T, self.q)
        H = [torch.zeros(T+1, self.p) for _ in range(self.k)]
        I = [torch.zeros(T, self.p) for _ in range(self.k)]
        F = [torch.zeros(T, self.p) for _ in range(self.k)]
        O = [torch.zeros(T, self.p) for _ in range(self.k)]
        C_n = [torch.zeros(T, self.p) for _ in range(self.k)] 
        C = [torch.zeros(T+1, self.p) for _ in range(self.k)]

        for t, x_t in enumerate(X):
            for l in range(self.k):
                temp = x_t if l == 0 else H[l-1][t+1]

                I[l][t] = torch.sigmoid(self.W_x_i[l]@temp+self.W_h_i[l]@H[l][t])
                F[l][t] = torch.sigmoid(self.W_x_f[l]@temp+self.W_h_f[l]@H[l][t])
                O[l][t] = torch.sigmoid(self.W_x_o[l]@temp+self.W_h_o[l]@H[l][t])
                C_n[l][t] = torch.tanh(self.W_x_c[l]@temp+self.W_h_c[l]@H[l][t])

                C[l][t+1] = F[l][t]*C[l][t]+I[l][t]*C_n[l][t]
                H[l][t+1] = O[l][t]*torch.tanh(C[l][t+1])
            Y[t] = self.W_y.t()@H[self.k-1][t+1]

        self.H = torch.stack([h[1:] for h in H])
        self.C = torch.stack([c[1:] for c in C])
        self.I, self.F, self.O, self.C_n = torch.stack(I), torch.stack(F), torch.stack(O), torch.stack(C_n)

        return Y
    

    def _backward_pass(self, X: torch.Tensor, Y: torch.tensor, Y_hat: torch.tensor):
        T = X.shape[0]

        dL_dh, dL_dc = torch.zeros_like(self.H), torch.zeros_like(self.C)
        dL_di, dL_df, dL_do, dL_dc_n = torch.zeros_like(self.I), torch.zeros_like(self.F), torch.zeros_like(self.O), torch.zeros_like(self.C_n)

        self.dL_dWx_i = [torch.zeros_like(w) for w in self.W_x_i]
        self.dL_dWx_f = [torch.zeros_like(w) for w in self.W_x_f]
        self.dL_dWx_o = [torch.zeros_like(w) for w in self.W_x_o]
        self.dL_dWx_c = [torch.zeros_like(w) for w in self.W_x_c]

        self.dL_dWh_i = [torch.zeros_like(w) for w in self.W_h_i]
        self.dL_dWh_f = [torch.zeros_like(w) for w in self.W_h_f]
        self.dL_dWh_o = [torch.zeros_like(w) for w in self.W_h_o]
        self.dL_dWh_c = [torch.zeros_like(w) for w in self.W_h_c]

        dL_dy = Y - Y_hat

        dL_dh[-1] = dL_dy@self.W_y.t()
        self.dL_dWy = self.H[-1].t()@dL_dy

        for t in range(T):
            for l in range(self.k-1, -1, -1):
                dL_dc[l][t] = dL_dh[l][t]*self.O[l][t]*(1-torch.tanh(self.C[l][t])**2)
        
                dL_di[l][t] = dL_dc[l][t]*self.C_n[l][t]
                dL_dc_n[l][t] = dL_dc[l][t]*self.I[l][t]
                dL_df[l][t] = dL_dc[l][t]*self.C[l][t]
                dL_do[l][t] = dL_dh[l][t]*torch.tanh(self.C[l][t])

                e_i = dL_di[l][t]*self.I[l][t]*(1-self.I[l][t])
                e_f = dL_df[l][t]*self.F[l][t]*(1-self.F[l][t])
                e_o = dL_do[l][t]*self.O[l][t]*(1-self.O[l][t])
                e_c = dL_dc_n[l][t]*(1-self.C_n[l][t]**2)

                if t > 0:
                    h_prev = self.H[l][t-1].view(1,-1)
                else:
                    h_prev = torch.zeros_like(self.H[l][0]).view(1,-1)

                self.dL_dWh_i[l] += e_i.view(-1,1)@h_prev
                self.dL_dWh_f[l] += e_f.view(-1,1)@h_prev
                self.dL_dWh_o[l] += e_o.view(-1,1)@h_prev
                self.dL_dWh_c[l] += e_c.view(-1,1)@h_prev

                if l >= 1:
                    dL_dh[l-1] = e_i@self.W_x_i[l]+e_f@self.W_x_f[l]+e_o@self.W_x_o[l]+e_c@self.W_x_c[l]
                
                    h_prev_l = self.H[l-1][t].view(1,-1)
                else:
                    h_prev_l = X[t].view(1,-1)

                self.dL_dWx_i[l] += e_i.view(-1,1)@h_prev_l
                self.dL_dWx_f[l] += e_f.view(-1,1)@h_prev_l
                self.dL_dWx_o[l] += e_o.view(-1,1)@h_prev_l
                self.dL_dWx_c[l] += e_c.view(-1,1)@h_prev_l

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
                self.W_x_i[l] -= a*self.dL_dWx_i[l]
                self.W_x_f[l] -= a*self.dL_dWx_f[l]
                self.W_x_o[l] -= a*self.dL_dWx_o[l]
                self.W_x_c[l] -= a*self.dL_dWx_c[l]

                self.W_h_i[l] -= a*self.dL_dWh_i[l]
                self.W_h_f[l] -= a*self.dL_dWh_f[l]
                self.W_h_o[l] -= a*self.dL_dWh_o[l]
                self.W_h_c[l] -= a*self.dL_dWh_c[l]
            
            self.W_y -= a*self.dL_dWy

            loss = torch.mean((Y - Y_hat) ** 2).item()
            if verbose:
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.6f}")
                