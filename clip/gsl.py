from torch.autograd import Function

class GradientScaleLayer(Function):

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.scale
        return grad_output, None

def gradient_scale_layer(x, scale):
    return GradientScaleLayer.apply(x, scale)
