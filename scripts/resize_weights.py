import torch

num_class = 8

pretrained_weights = torch.load("/home/cse/msr/siy207566/MaskTrackRCNN/checkpoints/mask_rcnn_r50_fpn_1x_20181010-069fa190.pth")

#print(pretrained_weights)

pretrained_weights['state_dict']['mask_head.conv_logits.weight'].resize_(num_class,256,1,1) 
pretrained_weights['state_dict']['mask_head.conv_logits.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.fc_cls.weight'].resize_(num_class, 1024)
pretrained_weights['state_dict']['bbox_head.fc_cls.bias'].resize_(num_class)
pretrained_weights['state_dict']['bbox_head.fc_reg.weight'].resize_(num_class*4, 1024)
pretrained_weights['state_dict']['bbox_head.fc_reg.bias'].resize_(num_class*4) 

torch.save(pretrained_weights,"/home/cse/msr/siy207566/MaskTrackRCNN/checkpoints/resized_weights.pth")
