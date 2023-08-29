import data_process as dls
import torch
from get_argparse import set_args
import train


def initial_process(HSI, Label, param):
    if_mlp_train = True
    if_feed_mlp = False
    if_sam_train = False
    if_feed_sam = True
    out_channel = 3

    height, width, band = HSI.shape
    if not if_feed_mlp:
        band = 3
        HSI = dls.set_pca(HSI, choose_band=band)
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

    for per_class in range(param.max_classes):
        per_input_label, per_loss_label, per_train_label, per_test_label \
            = input_label[per_class], loss_label[per_class], train_label[per_class], test_label[per_class]

        per_loss_label = torch.from_numpy(per_loss_label).to(param.device)
        per_test_label = torch.from_numpy(per_test_label).to(param.device)

        training = train.train(in_channel=band,
                               out_channel=out_channel,
                               param=param,
                               input_label=per_input_label,
                               loss_label=per_loss_label,
                               train_label=per_train_label,
                               test_label=per_test_label,
                               input_location=input_direct,
                               loss_location=loss_direct,
                               train_location=train_direct,
                               test_location=test_direct,
                               current_class=per_class)
        training.train_mode(mlp_train=if_mlp_train, sam_train=if_sam_train)
        training.feed_net(mlp_train=if_feed_mlp, sam_train=if_feed_sam)
        training.train_process(HSI.copy(), Label.copy())


if __name__ == "__main__":
    HSI_name = "PaviaU"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    initial_process(HSI, Label, args)
