"""NCR's two MLPs (DiffNet, SpecNet) as graphs the Kirin NPU can run through NNRt / HiAI Foundation.

The hash-grid encodings are not part of any graph: the GPU looks them up and writes them into the
network input, and takes the gradient of the encoding channels back from the training graph.

What the graphs reproduce (tiny-cuda-nn, see Source/RenderPasses/NeuralRadiosity/Model/Model.cu):
  - FullyFusedMLP: no biases, ReLU hidden layers, the output padded to 16 channels,
    SquarePlus output activation with K_ACT = 10.
  - RelativeL2Luminance loss over the first NCR_OUTPUT_DIM (4) channels, with the luminance of the
    prediction held constant in the denominator, and the gradient scaled by loss_scale = 128 (fp16).

NPU constraints (skills ohos-cann-kit / ohos-vulkan-npu-port): matrix products are 1x1 convolutions,
no compare-and-cast, weights are graph inputs. Weight tensors are laid out as conv kernels
(Cout, Cin, 1, 1), which is also the row-major order of tiny-cuda-nn's weight matrices.
A batch of N samples is a (1, C, H, W) tensor with H * W = N: channel-planar (NCHW).
"""
import torch
from torch import nn
import torch.nn.functional as F

K_ACT = 10.0          # tiny-cuda-nn SquarePlus scale
LOSS_EPS = 0.01       # RelativeL2Luminance: luminance^2 + 0.01
LOSS_SCALE = 128.0    # tcnn::default_loss_scale<__half>()
OUTPUT_DIM = 4        # NCR_OUTPUT_DIM: channels supervised by the loss
OUTPUT_PAD = 16       # FullyFusedMLP pads the output layer to 16
MASK_CHANNEL = 15     # channel of the target tensor that says whether the sample counts (1) or is padding (0)
MASK_EPS = 1e-4       # ReLU derivative as a / (a + eps); representable in fp16


class NetSpec:
    def __init__(self, name, input_dim, width, hidden_layers, encoding_dim):
        self.name = name
        self.input_dim = input_dim          # padded network input (multiple of 16)
        self.width = width
        self.hidden_layers = hidden_layers  # NCR_*_N_HIDDEN_LAYERS
        self.encoding_dim = encoding_dim    # leading input channels written by the hash grids

    def layer_shapes(self, output_channels=OUTPUT_PAD):
        """(Cout, Cin) of every weight matrix, in tiny-cuda-nn's parameter order."""
        shapes = [(self.width, self.input_dim)]
        shapes += [(self.width, self.width)] * (self.hidden_layers - 1)
        shapes.append((output_channels, self.width))
        return shapes

    def n_params(self):
        return sum(o * i for o, i in self.layer_shapes())


# NCRConfig.h: primary grid 6 levels x 2 features; cluster grid 4 clusters x 4 features.
DIFF = NetSpec("diff", input_dim=32, width=64, hidden_layers=3, encoding_dim=12)
SPEC = NetSpec("spec", input_dim=64, width=64, hidden_layers=4, encoding_dim=12 + 16)
NETS = {"diff": DIFF, "spec": SPEC}


SQUAREPLUS_RANGE = 64.0   # the square is taken of q / 64, see squareplus()


def squareplus(z):
    """Returns y = squareplus(z) = (q + sqrt(q^2 + 4)) / (2 K) with q = K z, and dy/dz.
    The NPU computes in fp16 (largest value 65504), where q^2 overflows from z = 25.6 on, a radiance
    HDR scenes do reach. So the root is taken as 64 * sqrt((q / 64)^2 + 4 / 64^2), which is the same
    number and holds up to z = 1638. The derivative is tiny-cuda-nn's y'^2 / (y'^2 + 1) with
    y' = K y, written as 0.5 * (1 + q / s) for the same reason."""
    q = z * K_ACT
    r = q * (1.0 / SQUAREPLUS_RANGE)
    s = torch.sqrt(r * r + 4.0 / (SQUAREPLUS_RANGE * SQUAREPLUS_RANGE)) * SQUAREPLUS_RANGE
    y = (q + s) * (0.5 / K_ACT)
    dy_dz = q / s * 0.5 + 0.5
    return y, dy_dz


def relu_mask(a, mode):
    """Derivative of ReLU from its output a >= 0, without a comparison (compare-and-cast falls off the NPU).
    "sign" is exact. "clip" is exact for a >= 2^-12 and linear below. "div" (the form Device MRRC runs
    on the NPU) is a / (a + eps): about 1% low at a = 0.01, which shows up as a few percent in single
    elements of g_x. Which of them stay on the NPU is for the probe to tell."""
    if mode == "sign":
        return torch.sign(a)
    if mode == "clip":
        # Two steps of 64 so that no intermediate value overflows fp16 (a * 4096 would above a = 16).
        return torch.clamp(torch.clamp(a * 64.0, 0.0, 1.0) * 64.0, 0.0, 1.0)
    return a / (a + MASK_EPS)


def batch_outer_sum(a, g):
    """sum_n g[:, n] a[:, n]^T for a (1, Cin, N, 1), g (1, Cout, N, 1) -> kernel layout (Cout, Cin, 1, 1).
    The batch becomes the input channels of one 1x1 convolution."""
    cin, n = a.shape[1], a.shape[2]
    cout = g.shape[1]
    a_t = a.permute(0, 2, 1, 3)                 # (1, N, Cin, 1)
    kernel = g.reshape(cout, n, 1, 1)           # (Cout, N, 1, 1)
    return F.conv2d(a_t, kernel).reshape(cout, cin, 1, 1)


class Inference(nn.Module):
    """x (1, Cin, H, W) with the encodings already written; w0..wk -> y (1, Cout, H, W)."""

    def __init__(self, spec, height, width, output_channels=OUTPUT_PAD):
        super().__init__()
        self.spec = spec
        self.shapes = spec.layer_shapes(output_channels)
        self.inputs = [torch.rand(1, spec.input_dim, height, width)]
        self.input_names = ["x"]
        for i, (cout, cin) in enumerate(self.shapes):
            self.inputs.append((torch.rand(cout, cin, 1, 1) - 0.5) * 0.5)
            self.input_names.append(f"w{i}")
        self.output_names = ["y"]

    def forward(self, inputs):
        h = inputs[0]
        weights = inputs[1:]
        for w in weights[:-1]:
            h = F.relu(F.conv2d(h, w))
        y, _ = squareplus(F.conv2d(h, weights[-1]))
        return y


class Training(nn.Module):
    """One training step's gradients, written by hand (the NPU has no autograd).
    Inputs: w0..wk; x (1, Cin, N, 1); target (1, 16, N, 1): the radiance in channels 0..OUTPUT_DIM-1
    and in channel MASK_CHANNEL a weight, 1 for a sample and 0 for padding. A batch is cut into
    chunks of N samples, the graph's fixed size, and the rows of the last chunk beyond the batch
    must not contribute.
    Outputs: g_w0..g_wk in the weight layout, and g_x (1, Cin, N, 1): the loss gradient with respect
    to the network input, of which the GPU scatters the encoding channels into the hash grids.
    The loss is normalised by this chunk's N; the caller rescales by N / (rows of the whole batch).
    All gradients carry the factor LOSS_SCALE, as in tiny-cuda-nn; the optimizer divides it out."""

    def __init__(self, spec, batch, mask="div"):
        super().__init__()
        self.spec = spec
        self.batch = batch
        self.mask = mask
        self.shapes = spec.layer_shapes()
        self.inputs, self.input_names, self.output_names = [], [], []
        for i, (cout, cin) in enumerate(self.shapes):
            self.inputs.append((torch.rand(cout, cin, 1, 1) - 0.5) * 0.5)
            self.input_names.append(f"w{i}")
            self.output_names.append(f"g_w{i}")
        target = torch.rand(1, OUTPUT_PAD, batch, 1)
        target[:, MASK_CHANNEL] = (torch.rand(batch, 1) > 0.2).float()
        self.inputs += [torch.rand(1, spec.input_dim, batch, 1), target]
        self.input_names += ["x", "target"]
        self.output_names.append("g_x")

        luminance = torch.zeros(1, OUTPUT_PAD, 1, 1)
        luminance[0, :3, 0, 0] = torch.tensor([0.299, 0.587, 0.114])
        supervised = torch.zeros(1, OUTPUT_PAD, 1, 1)
        supervised[0, :OUTPUT_DIM] = 1.0
        mask = torch.zeros(1, OUTPUT_PAD, 1, 1)
        mask[0, MASK_CHANNEL] = 1.0
        self.register_buffer("luminance_kernel", luminance)
        self.register_buffer("supervised", supervised)
        self.register_buffer("mask_kernel", mask)

    def forward(self, inputs):
        n_layers = len(self.shapes)
        weights = inputs[:n_layers]
        x, target = inputs[n_layers], inputs[n_layers + 1]

        activations = [x]            # activations[i] is the input of layer i
        h = x
        for w in weights[:-1]:
            h = F.relu(F.conv2d(h, w))
            activations.append(h)
        y, dy_dz = squareplus(F.conv2d(h, weights[-1]))

        luminance = F.conv2d(y, self.luminance_kernel)                 # (1, 1, N, 1)
        scale = 2.0 * LOSS_SCALE / (self.batch * OUTPUT_DIM)
        mask = F.conv2d(target, self.mask_kernel)                      # (1, 1, N, 1): picks the weight channel
        g = (y - target) * self.supervised * scale / (luminance * luminance + LOSS_EPS) * mask
        g = g * dy_dz                                                  # dL/dz of the output layer

        gradients = [None] * n_layers
        g_x = None
        for i in reversed(range(n_layers)):
            gradients[i] = batch_outer_sum(activations[i], g)
            g_in = F.conv2d(g, weights[i].permute(1, 0, 2, 3))         # back through layer i
            if i > 0:
                g = g_in * relu_mask(activations[i], self.mask)
            else:
                g_x = g_in
        return gradients + [g_x]


def reference_gradients(spec, weights, x, target):
    """The same step with autograd, following tiny-cuda-nn's forward and loss literally."""
    weights = [w.detach().clone().requires_grad_(True) for w in weights]
    x = x.detach().clone().requires_grad_(True)
    h = x
    for w in weights[:-1]:
        h = F.relu(F.conv2d(h, w))
    q = F.conv2d(h, weights[-1]) * K_ACT
    y = 0.5 * (q + torch.sqrt(q * q + 4.0)) / K_ACT
    luminance = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]).detach()
    n_total = x.shape[2] * x.shape[3] * OUTPUT_DIM
    difference = y[:, :OUTPUT_DIM] - target[:, :OUTPUT_DIM]
    weight = target[:, MASK_CHANNEL:MASK_CHANNEL + 1]
    loss = LOSS_SCALE * (weight * difference * difference / (luminance * luminance + LOSS_EPS)).sum() / n_total
    loss.backward()
    return [w.grad for w in weights] + [x.grad]
