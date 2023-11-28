import torch

mse_fn = torch.nn.MSELoss()
per_element_mse_fn = torch.nn.MSELoss(reduction="none")
per_element_bce_fn = torch.nn.BCEWithLogitsLoss(reduction="none")


def per_element_multi_scale_fn(
    model_output,
    gt,
    loss_name="mse",
    last_element=False,
):
    if loss_name == "mse":
        loss_fn = per_element_mse_fn
    elif loss_name == "bce":
        loss_fn = per_element_bce_fn

    N = gt.shape[0]
    gt = gt.reshape(N, -1)
    loss = [loss_fn(out.reshape(N, -1), gt) for out in model_output]

    loss = torch.stack(loss)

    return loss


def batch_multi_scale_fn(model_output, gt, loss_name="mse", use_resized=False):
    # if use_resized:
    #    loss = [(out - gt_img)**2 for out, gt_img in zip(model_output['model_out']['output'], gt['img'])]
    # else:
    per_element_multi_scale_mse = per_element_multi_scale_fn(
        model_output, gt, loss_name=loss_name
    )
    # Shape (batch_size,)
    return per_element_multi_scale_mse.view(gt.shape[0], -1).mean(dim=1)


def per_element_nll_fn(x, y):
    num_examples = x.size()[0]

    negative_log_likelihood = -(y * torch.log(x) + (1 - y) * torch.log(1 - x))

    return negative_log_likelihood


def per_element_rel_mse_fn(x, y, reduction=True):
    num_examples = x.size()[0]

    diff_norms = torch.norm(
        x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1
    )
    y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)

    return diff_norms / y_norms


def batch_mse_rel_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    # per_element_mse = per_element_mse_fn(x1, x2)
    per_element_mse = per_element_rel_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def batch_mse_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    per_element_mse = per_element_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


def batch_nll_fn(x1, x2):
    per_element_nll = per_element_nll_fn(x1, x2)
    return per_element_nll.view(x1.shape[0], -1).mean(dim=1)


def mse2psnr(mse):
    """Computes PSNR from MSE, assuming the MSE was calculated between signals
    lying in [0, 1].
    Args:
        mse (torch.Tensor or float):
    """
    return -10.0 * torch.log10(mse)


def psnr_fn(x1, x2):
    """Computes PSNR between signals x1 and x2. Note that the values of x1 and
    x2 are assumed to lie in [0, 1].
    Args:
        x1 (torch.Tensor): Shape (*).
        x2 (torch.Tensor): Shape (*).
    """
    return mse2psnr(mse_fn(x1, x2))


# from fno


# loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(
        self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True
    ):
        super(HsLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [
                1,
            ] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)
        return diff_norms / y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = (
            torch.cat(
                (
                    torch.arange(start=0, end=nx // 2, step=1),
                    torch.arange(start=-nx // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
        )
        k_y = (
            torch.cat(
                (
                    torch.arange(start=0, end=ny // 2, step=1),
                    torch.arange(start=-ny // 2, end=0, step=1),
                ),
                0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
        )
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced == False:
            weight = 1
            if k >= 1:
                weight += a[0] ** 2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(
                    k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4
                )
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        return loss
