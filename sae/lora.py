from peft.tuners.lora import Linear as LoraLinear


class LoraLinearWithHook(LoraLinear):
    def compute_lora_result(self, x):
        adapter_name = self.active_adapters[0]
        dropout = self.lora_dropout[adapter_name]
        lora_A_module = self.lora_A[adapter_name]
        lora_B_module = self.lora_B[adapter_name]
        scaling = self.scaling[adapter_name]

        h = dropout(x)
        out = lora_B_module(lora_A_module(h)) * scaling
        return out

    def forward(self, x):
        result = self.base_layer(x)
        lora_result = self.compute_lora_result(x)
        return result + lora_result


def replace_lora_linear(module):
    for name, child in module.named_children():
        if isinstance(child, LoraLinear):
            adapter_name = child.active_adapters[0]
            new_module = LoraLinearWithHook(
                base_layer=child.base_layer,
                adapter_name=adapter_name,
                in_features=child.in_features,
                out_features=child.out_features,
                r=child.r[adapter_name],
                lora_alpha=child.lora_alpha[adapter_name],
                lora_dropout=child.lora_dropout[adapter_name].p,
                fan_in_fan_out=child.fan_in_fan_out,
                bias=child.base_layer.bias is not None,
                init_lora_weights=False,
            )

            new_module.base_layer.weight = child.base_layer.weight
            if child.base_layer.bias is not None:
                new_module.base_layer.bias = child.base_layer.bias

            new_module.lora_A = child.lora_A
            new_module.lora_B = child.lora_B
            new_module.scaling = child.scaling
            setattr(module, name, new_module)
        else:
            replace_lora_linear(child)
