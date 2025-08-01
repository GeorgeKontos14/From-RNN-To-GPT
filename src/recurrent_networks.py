import math

import torch

from abc import ABC, abstractmethod

class BaseRNN(ABC):

    def __init__(self, d: int, p: int, k: int, q: int):
        self.d = d
        self.p = p
        self.k = k
        self.q = q

    @abstractmethod
    def _forward_pass(self, X: torch.Tensor):
        pass

    @abstractmethod
    def _backward_pass(self, X: torch.Tensor, Y: torch.Tensor, Y_hat: torch.Tensor):
        pass

    def fit(
        self,
        X: torch.Tensor,
        Y_hat: torch.Tensor,
        alpha: float = 0.1,
        epochs: int = 10,
        verbose: bool = False,
        decay: str = "const",
        decay_rate: float = 0.9
    ):
        history = []
        for ep in range(epochs):
            if decay == "exp":
                a = alpha * math.exp(-decay_rate * ep / epochs)
            elif decay == "inv":
                a = alpha / (1 + decay_rate * ep / epochs)
            else:
                a = alpha

            Y = self._forward_pass(X)
            self._backward_pass(X, Y, Y_hat)
            self._apply_gradients(a)

            loss = torch.mean((Y - Y_hat) ** 2).item()
            if verbose:
                print(f"Epoch {ep+1}/{epochs}, Loss: {loss:.6f}")
            history.append(loss)
        
        return history

    @abstractmethod
    def _apply_gradients(self, lr: float):
        pass

class RNN(BaseRNN):

    def __init__(self, d: int, p: int, k: int, q: int):
        super().__init__(d,p,k,q)

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

    def _apply_gradients(self, a: float):
        for l in range(self.k):
            self.W_x[l] -= a * self.dL_dWx[l]
            self.W_h[l] -= a * self.dL_dWh[l]
        self.W_y -= a * self.dL_dWy

class BiNN(BaseRNN):

    def __init__(self, d: int, p: int, k: int, q: int):
        super().__init__(d,p,k,q)

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

    def _apply_gradients(self, a: float):
        for l in range(self.k):
            self.W_x_f[l] -= a*self.dL_dWx_f[l]
            self.W_h_f[l] -= a*self.dL_dWh_f[l]
            self.W_x_b[l] -= a*self.dL_dWx_b[l]
            self.W_h_b[l] -= a*self.dL_dWh_b[l]
        
        self.W_y_f -= a*self.dL_dWy_f
        self.W_y_b -= a*self.dL_dWy_b

class LSTM(BaseRNN):

    def __init__(self, d: int, p: int, k: int, q: int):
        super().__init__(d,p,k,q)

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
                    dL_dh[l-1][t] = e_i@self.W_x_i[l]+e_f@self.W_x_f[l]+e_o@self.W_x_o[l]+e_c@self.W_x_c[l]
                
                    h_prev_l = self.H[l-1][t].view(1,-1)
                else:
                    h_prev_l = X[t].view(1,-1)

                self.dL_dWx_i[l] += e_i.view(-1,1)@h_prev_l
                self.dL_dWx_f[l] += e_f.view(-1,1)@h_prev_l
                self.dL_dWx_o[l] += e_o.view(-1,1)@h_prev_l
                self.dL_dWx_c[l] += e_c.view(-1,1)@h_prev_l

    def _apply_gradients(self, a: float):
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

class GRU(BaseRNN):

    def __init__(self, d: int, p: int, k: int, q: int):
        super().__init__(d,p,k,q)

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

    def _apply_gradients(self, a: float):
        for l in range(self.k):
            self.W_x_z[l] -= a*self.dL_dWx_z[l]
            self.W_x_r[l] -= a*self.dL_dWx_r[l]
            self.V_x[l] -= a*self.dL_dVx[l]

            self.W_h_z[l] -= a*self.dL_dWh_z[l]
            self.W_h_r[l] -= a*self.dL_dWh_r[l]
            self.V_h[l] -= a*self.dL_dVh[l]
        
        self.W_y -= a*self.dL_dWy