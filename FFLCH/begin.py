import random
import sys

import data_process as dls
import torch
from get_argparse import set_args
from net import FFLCHs
import numpy as np


def initial_process(HSI, Label, param):
    # TODO(Byan Xia): 怎么把这组参数另找个地方设置呢？
    if_mlp_train = True
    if_feed_mlp = True
    if_sam_train = False
    if_feed_sam = True
    out_channel = 3

    height, width, band = HSI.shape
    if not if_feed_mlp:
        band = 3
        HSI = dls.set_pca(HSI, choose_band=band)
    # TODO(Byan Xia): 感觉下面的数据处理部分可以继续优化，有时间去做做
    # Get class location
    input_location, loss_location, train_location, test_location \
        = dls.location_seg(Label, param.train_num, param.input_sam)
    # Get concated location
    input_direct, loss_direct, train_direct, test_direct \
        = dls.concat_dict(input_location), dls.concat_dict(loss_location), dls.concat_dict(train_location), dls.concat_dict(test_location)
    # Get lable
    input_label, loss_label, train_label, test_label \
        = dls.get_label(input_location), dls.get_label(loss_location), dls.get_label(train_location), dls.get_label(test_location)

    loss_direct = torch.from_numpy(loss_direct).to(param.device)
    test_direct = torch.from_numpy(test_direct).to(param.device)

    training = FFLCHs(in_channel=band,
                      out_channel=out_channel,
                      param=param,
                      input_location=input_direct,
                      loss_location=loss_direct,
                      train_location=train_direct,
                      test_location=test_direct
                      )
    for epoch in range(1, param.epochs + 1):
        per_class = random.randint(0, param.max_classes - 1)
        # Per class label
        per_input_label, per_loss_label, per_train_label, per_test_label \
            = input_label[per_class], loss_label[per_class], train_label[per_class], test_label[per_class]

        per_loss_label = torch.from_numpy(per_loss_label).to(param.device)
        per_test_label = torch.from_numpy(per_test_label).to(param.device)
        # set current class label
        training.set_label(input_label=per_input_label,
                           loss_label=per_loss_label,
                           train_label=per_train_label,
                           test_label=per_test_label,
                           current_class=per_class,
                           epoch=epoch)
        # Set train mode
        training.train_mode(mlp_train=if_mlp_train, sam_train=if_sam_train)
        training.feed_net(mlp_train=if_feed_mlp, sam_train=if_feed_sam)
        # Train entry
        training.per_class_training(HSI.copy(), Label.copy())

    torch_train_pic = torch.from_numpy(HSI.astype("int32")).to(param.device)  # Sample data process
    torch_train_pic = torch.unsqueeze(torch_train_pic.permute((2, 0, 1)), dim=0)
    # Initial AA can and correct can
    aa = np.zeros(shape=(param.max_classes,))
    sames = np.zeros(shape=(param.max_classes,))

    for classes in range(param.max_classes):
        # Per class label
        per_input_label, per_loss_label, per_train_label, per_test_label \
            = input_label[classes], loss_label[classes], train_label[classes], test_label[classes]

        per_loss_label = torch.from_numpy(per_loss_label).to(param.device)
        per_test_label = torch.from_numpy(per_test_label).to(param.device)
        # set current class label
        training.set_label(input_label=per_input_label,
                           loss_label=per_loss_label,
                           train_label=per_train_label,
                           test_label=per_test_label,
                           current_class=classes,
                           epoch=1)
        # predict(Test)
        per_acc, same, mlp_out, sam_out = training.predict(HSI.shape, torch_train_pic)
        aa[classes], sames[classes] = per_acc, same
        # Show sam outcome
        dls.show_pic(sam_out, train_direct, classes, param.train_num, 0, "final", replace=True)
    # AA
    AA = np.average(aa)
    # OA
    OA = np.sum(sames) / test_direct.shape[0] * 100
    print(f"AA is {str(AA)}, OA is {str(OA)}.")


if __name__ == "__main__":
    HSI_name = "PaviaU"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    initial_process(HSI, Label, args)
    sys.exit(114514)
