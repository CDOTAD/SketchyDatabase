import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_features, in_features//4, kernel_size=1),
            nn.Conv2d(in_features//4, 1, kernel_size=1)
        )

    def forward(self, x):

        # print('AttentionLayer.forward.x.size', x.size())

        attn_mask = self.net(x)
        # print('AttentionLayer.forward.attn_mask.size', attn_mask.size())
        attn_mask = attn_mask.view(attn_mask.size(0), -1)
        # print('AttentionLayer.forward.attn_mask.view.size', attn_mask.size())
        attn_mask = nn.Softmax(dim=1)(attn_mask)
        # print('AttentionLayer.forward.attn_mask.softmax.size', attn_mask.size())
        attn_mask = attn_mask.view(attn_mask.size(0), 1, x.size(2), x.size(3))
        # print('AttentionLayer.forward.attn_mask', attn_mask.size())
        x_attn = x * attn_mask
        # print('AttentionLayer.forward.x_attn', x_attn.size())
        x = x + x_attn
        # print('AttentionLayer.forward.x', x.size())

        return x, attn_mask

