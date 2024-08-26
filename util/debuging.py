import torch
import numpy as np

def check_nan(tensor, name):
    if tensor is None:
        print(f"{name} is None")
        return False
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
        return True
    return False

def check_inf(tensor, name):
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")
        return True
    return False

def print_stats(tensor, name):
    if tensor is None:
        print(f"{name} is None")
    else:
        print(f"{name} - min: {tensor.min().item():.5f}, max: {tensor.max().item():.5f}, mean: {tensor.mean().item():.5f}, std: {tensor.std().item():.5f}")

class NaNDebugger(object):
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def add_hooks(self):
        for name, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(self.forward_hook(name)))
            self.hooks.append(module.register_backward_hook(self.backward_hook(name)))

    def forward_hook(self, name):
        def hook(module, input, output):
            if check_nan(output, f"{name} (output)"):
                raise RuntimeError(f"NaN detected in output of {name}")
            print_stats(output, f"{name} (output)")
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            if grad_input is not None:
                for i, grad in enumerate(grad_input):
                    if grad is not None:
                        if check_nan(grad, f"{name} (grad_input[{i}])"):
                            raise RuntimeError(f"NaN detected in grad_input of {name}")
                        print_stats(grad, f"{name} (grad_input[{i}])")
                    else:
                        print(f"{name} (grad_input[{i}]) is None")
            else:
                print(f"{name} (grad_input) is None")

            if grad_output is not None:
                for i, grad in enumerate(grad_output):
                    if grad is not None:
                        if check_nan(grad, f"{name} (grad_output[{i}])"):
                            raise RuntimeError(f"NaN detected in grad_output of {name}")
                        print_stats(grad, f"{name} (grad_output[{i}])")
                    else:
                        print(f"{name} (grad_output[{i}]) is None")
            else:
                print(f"{name} (grad_output) is None")
        return hook

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def debug_train_step(model, inputs, labels, criterion, optimizer):
    debugger = NaNDebugger(model)
    debugger.add_hooks()

    try:
        # Forward pass
        outputs = model(inputs)
        if check_nan(outputs, "model output"):
            raise RuntimeError("NaN detected in model output")
        print_stats(outputs, "model output")

        # Loss computation
        loss = criterion(outputs, labels)
        if check_nan(loss, "loss"):
            raise RuntimeError("NaN detected in loss computation")
        print(f"Loss: {loss.item():.5f}")

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Check gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                if check_nan(param.grad, f"{name} (gradient)"):
                    raise RuntimeError(f"NaN detected in gradient of {name}")
                print_stats(param.grad, f"{name} (gradient)")

        # Gradient clipping
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        # Optimizer step
        optimizer.step()

        # Check weights after update
        for name, param in model.named_parameters():
            if check_nan(param, f"{name} (weight after update)"):
                raise RuntimeError(f"NaN detected in weight of {name} after update")
            print_stats(param, f"{name} (weight after update)")

    finally:
        debugger.remove_hooks()
