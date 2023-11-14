import argparse
import warnings
warnings.filterwarnings("ignore")

from train_duq_fm import train_model
from train_duq_cifar import train_model_cifar
from evaluate_ood import get_auroc_ood
from dataset import *
from models.models import MNIST_Net, Fashion_MNIST_Net, Cifar_10_Net




data_dic = {
    'MNIST': MNIST_dataset,
    'FashionMNIST': Fashion_MNIST_dataset, 
    'Cifar_10': Cifar_10_dataset,
    'SVHN': SVHN_dataset, 
    'Imagenet_r': TinyImagenet_r_dataset,
    'Imagenet_c': TinyImagenet_c_dataset
}


data_model = {
    'MNIST': MNIST_Net,
    'FashionMNIST': Fashion_MNIST_Net, 
    'Cifar_10': Cifar_10_Net   
}


def main():
    parser = argparse.ArgumentParser(description="DUQ parameters")

    # Add a positional argument for the number
    parser.add_argument("InD_Dataset", type=str, help="The name of the InD dataset.")
    parser.add_argument("train_batch_size", type=int, help="train_batch_size")
    parser.add_argument("test_batch_size", type=int, help="test_batch_size")
    # parser.add_argument("gpu", type=int, help="number of gpu")

    # Parse the command-line arguments
    args = parser.parse_args()


    train_set, test_set, trloader, tsloader = data_dic[args.InD_Dataset](batch_size = args.train_batch_size, 
                                                                    test_batch_size = args.test_batch_size)
    OOD_sets, OOD_loaders = [], []
    if args.InD_Dataset == 'Cifar_10':
        OOD_Dataset = ['SVHN', 'Imagenet_r', 'Imagenet_c']
    else:
        if args.InD_Dataset == 'MNIST':
            OOD_Dataset = ['FashionMNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']
        elif args.InD_Dataset == 'FashionMNIST':
            OOD_Dataset = ['MNIST', 'Cifar_10', 'SVHN', 'Imagenet_r', 'Imagenet_c']

    # Get all OOD datasets     
    for dataset in OOD_Dataset:
        _, OOD_set, _, OODloader = data_dic[dataset](batch_size = args.train_batch_size, 
                                                    test_batch_size = args.test_batch_size, into_grey = True)
        OOD_sets.append(OOD_set)
        OOD_loaders.append(OODloader)


    if args.InD_Dataset == 'Cifar_10':
        for i in range(len(OOD_Dataset)):
            print("OOD:", OOD_Dataset[i])
            train_model_cifar(train_set, test_set, OOD_sets[i])
    else:
        l_gradient_penalties = [0.01, 0.1, 0.3]
        length_scales = [0.05, 0.5, 1.0]   # 0.1, 0.3, 

        # l_gradient_penalties = [0.1]
        # length_scales = [0.05]

        repetition = 1  # Increase for multiple repetitions
        final_model = False  # set true for final model to train on full train set

        results = {}

        for l_gradient_penalty in l_gradient_penalties:
            for length_scale in length_scales:
                val_accuracies = []
                test_accuracies = []
                ood_accuracies = []
                roc_aucs_mnist = []
                # roc_aucs_notmnist = []

                for _ in range(repetition):
                    print(" ### NEW MODEL ### ")
                    
                    print("train with parameters:", l_gradient_penalty, length_scale)
                    model, val_accuracy, test_accuracy = train_model(
                        l_gradient_penalty, length_scale, final_model, train_set, test_set
                    )
                    # accuracy, roc_auc_mnist = get_fashionmnist_mnist_ood(model)
                    # _, roc_auc_notmnist = get_fashionmnist_notmnist_ood(model)

                    for i in range(len(OOD_Dataset)):
                        print("OOD: ", OOD_Dataset[i])
                        ind_accuracy, ood_accuracy, roc_auc = get_auroc_ood(test_set, OOD_sets[i], model, l_gradient_penalty, length_scale, OOD_Dataset[i])

                        # val_accuracies.append(val_accuracy)
                        # test_accuracies.append(test_accuracy)

                        # ood_accuracies.append(ood_accuracy)

                        # roc_aucs_mnist.append(roc_auc_mnist)
                        # roc_aucs_notmnist.append(roc_auc_notmnist)

                        ## TODO: why take the average instead of maximum here????
                        print("val_acc, test_acc, ind_acc, ood_acc, auc:")
                        print(val_accuracy, test_accuracy, ind_accuracy, ood_accuracy, roc_auc)

                        # results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                        #     (np.mean(val_accuracies), np.std(val_accuracies)),
                        #     (np.mean(test_accuracies), np.std(test_accuracies)),
                        #     (np.mean(ood_accuracies), np.std(ood_accuracies)),
                        #     (np.mean(roc_aucs_mnist), np.std(roc_aucs_mnist)),
                        # ]
                        # print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])

        # print(results)


if __name__ == "__main__":
    main()