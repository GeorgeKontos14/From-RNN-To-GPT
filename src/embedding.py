import math

import torch

class Word2Vec:

    def __init__(self, V: list[str], d: int, M: int, skipgram: bool = False):
        self.is_skipgram = skipgram

        self.word_ind_map = {w: i for i, w in enumerate(V)}
        self.ind_word_map = {i: w for i, w in enumerate(V)}
        self.V_size = len(V)
        self.d = d
        self.M = M

        self.W_in = torch.rand((self.V_size, d))
        self.W_out = torch.rand((d, self.V_size))

        self.dL_dW_in = torch.zeros_like(self.W_in)
        self.dL_dW_out = torch.zeros_like(self.W_out)

        self.h = torch.zeros(d)

    def _encode(self, w: str) -> torch.Tensor:
        ind = self.word_ind_map[w]
        x = torch.zeros(self.V_size)
        x[ind] = 1
        return x
    
    def _sample(self, P: torch.Tensor) -> torch.Tensor:
        ind = torch.argmax(P).item()
        y = torch.zeros(self.V_size)
        y[ind] = 1
        return y
    
    def _decode(self, y: torch.Tensor) -> str:
        ind = torch.where(y == 1)[0].item()
        return self.ind_word_map[ind]

    def _forward_pass(self, context_encoding:torch.Tensor, middle_encoding: torch.Tensor) -> torch.Tensor:
        if self.is_skipgram:
            self.h = middle_encoding@self.W_in
        else:
            V = context_encoding@self.W_in
            self.h = V.mean(dim=0)
        z = self.h.t()@self.W_out
        P = torch.softmax(z, dim=0)
        return P
    
    def _backward_pass(self, P: torch.Tensor, middle_ind: int, context_inds: list[int]):
        dL_dz = P.clone()
        
        if self.is_skipgram:
            self.dL_dW_in.fill_(0)
            dL_dz.fill_(0)
            
            for target_ind in context_inds:
                current_dL_dz = P.clone()
                current_dL_dz[target_ind] -= 1
                self.dL_dW_out += self.h.view(-1,1) @ current_dL_dz.view(1,-1)
                self.dL_dW_in[middle_ind] += self.W_out @ current_dL_dz

            if len(context_inds) > 0:
                self.dL_dW_out /= len(context_inds)
                self.dL_dW_in[middle_ind] /= len(context_inds)
                
        else:
            dL_dz[middle_ind] -= 1
            self.dL_dW_out = self.h.view(-1,1) @ dL_dz.view(1,-1)

            self.dL_dW_in.fill_(0)
            dL_dh = self.W_out @ dL_dz

            for ind in context_inds:
                self.dL_dW_in[ind] = dL_dh / len(context_inds)

    def embed(
            self, 
            corpus: list[list[str]],
            alpha: float = 0.1,
            epochs: int = 10,
            decay: str = "const",
            decay_rate: float = 0.9
        ) -> torch.Tensor:

        encodings = []
        middle_encodings = []
        context_inds = []
        middle_inds = []
        for window in corpus:
            current_contex_inds = []
            X = []
            for i, w in enumerate(window):
                ind = self.word_ind_map[w]
                if i == self.M:
                    middle_inds.append(ind)
                    middle_encodings.append(self._encode(w))
                else:
                    current_contex_inds.append(ind)
                    X.append(self._encode(w))
            context_inds.append(current_contex_inds)
            encodings.append(torch.stack(X))
        
        for ep in range(epochs):
            if decay == "exp":
                a = alpha * math.exp(-decay_rate * ep / epochs)
            elif decay == "inv":
                a = alpha / (1 + decay_rate * ep / epochs)
            else:
                a = alpha
            
            for j, X in enumerate(encodings):
                P = self._forward_pass(encodings[j], middle_encodings[j])
                self._backward_pass(P, middle_inds[j], context_inds[j])

                self.W_in -= a*self.dL_dW_in
                self.W_out -= a*self.dL_dW_out

        if self.is_skipgram:
            return self.W_in
        else:
            return self.W_out.t()
