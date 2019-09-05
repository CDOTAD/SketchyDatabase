import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, p=2, m=0.3):
        super(TripletLoss, self).__init__()

        self.p = p
        self.m = m

    def forward(self, sketch, photo):

        bs = sketch.size(0)

        pos_distance = sketch - photo
        pos_distance = torch.pow(pos_distance, 2)
        pos_distance = torch.sqrt(torch.sum(pos_distance, dim=1))

        sketch_self = sketch.unsqueeze(0)
        photo_T = photo.unsqueeze(1)

        negative_distance = sketch_self - photo_T
        negative_distance = torch.pow(negative_distance, 2)
        negative_distance = torch.sqrt(torch.sum(negative_distance, dim=2))

        triplet_loss = pos_distance - negative_distance   # bs x bs x num_vec
        # print('TripletLoss.forward.triplet_loss', triplet_loss)
        triplet_loss = triplet_loss + self.m
        eye = torch.eye(bs).cuda()
        triplet_loss = triplet_loss * (1 - eye)
        triplet_loss = F.relu(triplet_loss)

        triplet_loss = torch.sum(triplet_loss, dim=1)

        return torch.sum(triplet_loss)
