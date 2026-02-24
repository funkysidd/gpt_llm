import torch
import tiktoken

from logger import Logging, LogLevel
from gelu import GELU


class ShortcutConnections(torch.nn.Module):
    def __init__(self, sizes, use_shortcut: bool):
        super().__init__()

        self.use_shortcut = use_shortcut
        self.layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(sizes[i], sizes[i + 1]), GELU()) for i in range(0, len(sizes) - 1)]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and layer_output.shape == x.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x


def print_gradients(model: ShortcutConnections, x, target):
    logits = model(x)

    loss = torch.nn.MSELoss()
    loss = loss(logits, target)

    loss.backward()

    for name, param in model.named_parameters():
        if "weight" in name:
            Logging.log(
                LogLevel.INFO,
                f"{name} has gradient mean of {param.grad.abs().mean().item()}",
            )


if __name__ == "__main__":
    Logging.set_log_level(LogLevel.INFO)

    # The last layer transforms the incoming tensor from number of dimensions 3 to 1, so we need to be exact in
    # specifying the resulting tensor dimension i.e., (1, 1) as opposed to simply 1.
    sizes = [3, 3, 3, 3, 3, 1]
    x = torch.tensor([[1.0, 0.0, -1.0]])  # Dimension: (1, 3)
    target = torch.tensor([[0.0]])  # Dimension: (1, 1)

    Logging.log(LogLevel.INFO, "Testing with shortcut connections: False")
    torch.manual_seed(123)
    model_without_shortcuts = ShortcutConnections(sizes, use_shortcut=False)
    print_gradients(model_without_shortcuts, x, target)

    Logging.log(LogLevel.INFO, "Testing with shortcut connections: True")
    torch.manual_seed(123)
    model_with_shortcuts = ShortcutConnections(sizes, use_shortcut=True)
    print_gradients(model_with_shortcuts, x, target)
