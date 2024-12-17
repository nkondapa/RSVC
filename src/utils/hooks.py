import torch


class ActivationHook:

    def __init__(self, move_to_cpu_in_hook=False, move_to_cpu_every=None):
        self.layer_activations = {}
        self.handles = []
        self.layer_names = None
        self.move_to_cpu_in_hook = move_to_cpu_in_hook
        self.move_to_cpu_every = move_to_cpu_every
        self.layer_gpu_indices = {}
        self.layer_count = {}

    def hook_fn_factory(self, layer_name, post_activation_fn=None):
        def hook_fn(module, input, output):
            if self.move_to_cpu_in_hook:
                x = output.detach().clone().cpu()
            else:
                x = output.detach().clone()

            if post_activation_fn is not None:
                x = post_activation_fn(x)
            self.layer_activations[layer_name].append(x)

            self.layer_gpu_indices[layer_name].append(self.layer_count[layer_name])
            self.layer_count[layer_name] += 1

            # only moves if needed (according to move_to_cpu_every param)
            self.move_to_cpu(layer_name)

        return hook_fn

    def move_to_cpu(self, layer_name, finish=False):
        if self.move_to_cpu_every and (len(self.layer_gpu_indices[layer_name]) == self.move_to_cpu_every or finish):
            for i in self.layer_gpu_indices[layer_name]:
                self.layer_activations[layer_name][i] = self.layer_activations[layer_name][i].cpu()
            self.layer_gpu_indices[layer_name] = []

    def register_hooks(self, layer_names, layer_modules, post_activation_fn=None):

        assert len(layer_modules) == len(layer_names)

        self.layer_names = layer_names
        for name in layer_names:
            self.layer_activations[name] = []
            self.layer_gpu_indices[name] = []
            self.layer_count[name] = 0

        for name, module in zip(self.layer_names, layer_modules):
            hook_fn = self.hook_fn_factory(name, post_activation_fn=post_activation_fn)
            handle = module.register_forward_hook(hook_fn)
            self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def concatenate_layer_activations(self):
        for name in self.layer_names:
            self.move_to_cpu(name, finish=True)
            self.layer_activations[name] = torch.cat(self.layer_activations[name], dim=0).cpu()

    def reset_activation_dict(self):
        for name in self.layer_names:
            self.layer_activations[name] = []
            self.layer_gpu_indices[name] = []
            self.layer_count[name] = 0