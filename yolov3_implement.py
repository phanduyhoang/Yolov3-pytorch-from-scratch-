import torch
import torch.nn as nn

config = [
    (32, 3, 1),
    (64, 3, 2),
    ["B", 1],
    (128, 3, 2),
    ["B", 2],
    (256, 3, 2),
    ["B", 8],
    (512, 3, 2),
    ["B", 8],
    (1024, 3, 2),
    ["B", 4],  # To this point
    (512, 1, 1),
    (1024, 3, 1),
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_repeats):
            self.layers.append(nn.Sequential(
                CNNBlock(channels, channels // 2, kernel_size=1),
                CNNBlock(channels // 2, channels, kernel_size=3, padding=1)
            ))
        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = layer(x) + x
            else:
                x = layer(x)
        return x

class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
            .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
            .permute(0, 1, 3, 4, 2)
        )

class YOLOv3(nn.Module):
    def __init__(self, in_channels=3, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.layers = self._create_conv_layers()

    def forward(self, x):
        outputs = []
        route_connections = []
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, ScalePrediction):
                output = layer(x)
                outputs.append(output)
                print(f"Layer {idx}: ScalePrediction output shape: {output.shape}")
                continue

            x = layer(x)
            print(f"Layer {idx}: After {layer.__class__.__name__}, shape: {x.shape}")

            if isinstance(layer, ResidualBlock) and layer.num_repeats == 8:
                route_connections.append(x)
                print(f"Layer {idx}: Added to route_connections, current connections: {[t.shape for t in route_connections]}")
            elif isinstance(layer, nn.Upsample):
                if not route_connections:
                    raise ValueError("No route connections available for concatenation.")
                route = route_connections.pop()
                x = torch.cat([x, route], dim=1)
                print(f"Layer {idx}: After concatenation, shape: {x.shape}")
        return outputs

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for module in config:
            if isinstance(module, tuple):
                out_channels, kernel_size, stride = module
                layers.append(
                    CNNBlock(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=1 if kernel_size == 3 else 0,
                    )
                )
                in_channels = out_channels

            elif isinstance(module, list):
                num_repeats = module[1]
                layers.append(ResidualBlock(in_channels, num_repeats=num_repeats))

            elif isinstance(module, str):
                if module == 'S':
                    layers += [
                        ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                        CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                        ScalePrediction(in_channels // 2, num_classes=self.num_classes)
                    ]
                    in_channels = in_channels // 2
                elif module == 'U':  # Correctly handle "U" within the string check
                    layers.append(nn.Upsample(scale_factor=2))
                    in_channels = in_channels * 3

        return layers

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = YOLOv3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)  # Single forward pass

    # Use the stored output for all assertions
    try:
        assert out[0].shape == (2, 3, IMAGE_SIZE // 32, IMAGE_SIZE // 32, num_classes + 5), f"Expected out[0] shape: {(2,3,13,13,25)}, but got: {out[0].shape}"
        assert out[1].shape == (2, 3, IMAGE_SIZE // 16, IMAGE_SIZE // 16, num_classes + 5), f"Expected out[1] shape: {(2,3,26,26,25)}, but got: {out[1].shape}"
        assert out[2].shape == (2, 3, IMAGE_SIZE // 8, IMAGE_SIZE // 8, num_classes + 5), f"Expected out[2] shape: {(2,3,52,52,25)}, but got: {out[2].shape}"
        print('success')
    except AssertionError as e:
        print(e)
