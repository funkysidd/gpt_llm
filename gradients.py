from logger import Logging, LogLevel

import torch
import torch.nn.functional as F
from torch.autograd import grad

def grad_manual():
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b 
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)

    grad_loss_w1 = grad(loss, w1, retain_graph=True)
    grad_loss_b = grad(loss, b, retain_graph=False) # Frees the computational graph beyond this operation

    Logging.log(LogLevel.INFO, "a : {}".format(a))
    Logging.log(LogLevel.INFO, "loss : {}".format(loss))
    Logging.log(LogLevel.INFO, "grad_loss_w1 : {}".format(grad_loss_w1))
    Logging.log(LogLevel.INFO, "grad_loss_b : {}".format(grad_loss_b))

def grad_auto():
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b 
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)

    loss.backward()

    Logging.log(LogLevel.INFO, "a : {}".format(a))
    Logging.log(LogLevel.INFO, "loss : {}".format(loss))
    Logging.log(LogLevel.INFO, "grad_loss_w1 : {}".format(w1.grad))
    Logging.log(LogLevel.INFO, "grad_loss_b : {}".format(b.grad))

if __name__ == '__main__':
    Logging.set_log_level(LogLevel.INFO)

    grad_manual()
    grad_auto()
    