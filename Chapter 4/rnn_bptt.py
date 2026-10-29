import torch
import torch.nn as nn

torch.manual_seed(0)


class SimpleRNN(nn.Module):
    """h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h);  y = W_hy h_T + b_y"""

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # These SAME parameters are reused at every time step (parameter sharing)
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.5)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.5)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        self.W_hy = nn.Parameter(torch.randn(output_size, hidden_size) * 0.5)
        self.b_y = nn.Parameter(torch.zeros(output_size))

    def forward(self, x):                      # x: (T, input_size)
        h = torch.zeros(self.hidden_size)      # h_0
        hs = []
        for t in range(x.shape[0]):
            h = torch.tanh(self.W_xh @ x[t] + self.W_hh @ h + self.b_h)
            h.retain_grad()                    # keep dL/dh_t for inspection
            hs.append(h)                       # stored for the backward pass
        y = self.W_hy @ h + self.b_y           # loss depends on final h_T only
        return y, hs


T, D_in, D_h, D_out = 6, 3, 4, 1
model = SimpleRNN(D_in, D_h, D_out)
x = torch.randn(T, D_in)
target = torch.tensor([1.0])

y, hs = model(x)
loss = 0.5 * ((y - target) ** 2).sum()         # scalar loss
loss.backward()                                # BPTT

print(f"Loss = {loss.item():.6f}\n")
print("=== Parameter gradients ===")
for name, p in model.named_parameters():
    print(f"{name}  shape={tuple(p.shape)}\n{p.grad}\n")

print("=== dL/dh_t (flowing backward through time) ===")
for t, h in enumerate(hs, start=1):
    print(f"t={t}: {h.grad}   |grad|={h.grad.norm().item():.6f}")

# Verify the chain rule manually: dL/dh_1 = dL/dh_T * prod_{k=2..T} dh_k/dh_{k-1}
# dh_k/dh_{k-1} = diag(1 - h_k^2) @ W_hh
with torch.no_grad():
    grad = hs[-1].grad.clone()                 # dL/dh_T
    for k in range(T - 1, 0, -1):              # go from h_T back to h_1
        J = torch.diag(1 - hs[k] ** 2) @ model.W_hh
        grad = J.T @ grad
print("\nManual chain-rule dL/dh_1:", grad)
print("Autograd dL/dh_1:         ", hs[0].grad)
