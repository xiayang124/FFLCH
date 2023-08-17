from data_process import data_aug, set_pca, show_pic
from get_argparse import set_args
from train import train


def train_process(HSI, Label, param):
    aug = data_aug(HSI, Label, param.max_classes, param.train_num)
    # Get train location and test location
    train_location, test_location = aug.segment_data()
    # Set test data args
    aug.set_test_data_num()

    height, width, band = HSI.shape
    # Seg to 3 band data
    pca_HSI = set_pca(HSI, choose_band=3)

    trains = train(in_channel=3, out_channel=3, train_location=train_location, test_location=test_location, param=param)
    trains.train_mode()
    trains.feed_net(sam_train=True, mlp_train=False)
    trains.train_process(pca_HSI, Label)


if __name__ == "__main__":
    HSI_name = "Pavia"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    # debug
    args.epochs = 500

    train_process(HSI, Label, args)
