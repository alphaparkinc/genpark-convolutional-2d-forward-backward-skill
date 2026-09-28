"""2D Spatial Convolution Layer.
100% Python Standard Library.
"""

class Conv2D:
    """2D spatial convolution operation with configurable stride and zero-padding."""

    @staticmethod
    def forward(image: list, kernel: list, stride: int = 1, padding: int = 0) -> list:
        h = len(image)
        w = len(image[0])
        kh = len(kernel)
        kw = len(kernel[0])

        if padding > 0:
            padded = [[0.0] * (w + 2 * padding) for _ in range(h + 2 * padding)]
            for r in range(h):
                for c in range(w):
                    padded[r + padding][c + padding] = image[r][c]
            image = padded
            h += 2 * padding
            w += 2 * padding

        out_h = (h - kh) // stride + 1
        out_w = (w - kw) // stride + 1
        out = [[0.0] * out_w for _ in range(out_h)]

        for r in range(out_h):
            for c in range(out_w):
                val = 0.0
                for kr in range(kh):
                    for kc in range(kw):
                        val += image[r * stride + kr][c * stride + kc] * kernel[kr][kc]
                out[r][c] = val
        return out
