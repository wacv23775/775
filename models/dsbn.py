from torch import nn


class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(self, num_features, num_classes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(_DomainSpecificBatchNorm, self).__init__()
        #         self.bns = nn.ModuleList([nn.modules.batchnorm._BatchNorm(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input):
        raise NotImplementedError

    def forward(self, x, domain_label):
        self._check_input_dim(x)
        bn = self.bns[domain_label[0]]
        return bn(x), domain_label


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class DomainSpecificBatchNorm3d(_DomainSpecificBatchNorm):
    def __init__(self, num_features, num_classes=2, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(DomainSpecificBatchNorm3d, self).__init__(num_features, num_classes, eps=eps, momentum=momentum, affine=affine,
                 track_running_stats=track_running_stats)
        self.bns = nn.ModuleList(
            [nn.BatchNorm3d(num_features, eps, momentum, affine, track_running_stats) for _ in range(num_classes)])

    def _check_input_dim(self, input):
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))