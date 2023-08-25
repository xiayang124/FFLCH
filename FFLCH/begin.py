from data_process import set_pca, segment_data
from get_argparse import set_args
from train import train


def train_process(HSI, Label, param):
    # Get train location and test location
    train_location, test_location = segment_data(HSI, Label, param.train_num)
    # Seg to 3 band data
    pca_HSI = set_pca(HSI, choose_band=3)

    trains = train(in_channel=3, out_channel=3, param=param)
    trains.train_mode()
    trains.feed_net(sam_train=True, mlp_train=True)
    # trains.all_generate(pca_HSI)
    trains.train_process(pca_HSI, Label, train_location, test_location, param)


if __name__ == "__main__":
    HSI_name = "PaviaU"

    # Get the args and data
    arg = set_args(HSI_name)
    args, HSI, Label = arg.get_arg()

    # debug
    args.epochs = 500

    train_process(HSI, Label, args)
