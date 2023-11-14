import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# from utils.datasets import (
#     get_CIFAR10,
#     get_SVHN,
#     get_FashionMNIST,
#     get_MNIST,
#     get_notMNIST,
# )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_ood_datasets(true_dataset, ood_dataset):
    # # Preprocess OoD dataset same as true dataset
    # ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]
    # print(true_dataset[0][0].size(), ood_dataset[0][0].size())

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)
    # print(concat_datasets[0])

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, dataloader):
    model.eval()

    with torch.no_grad():
        scores = []
        accuracies = []
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            kernel_distance, pred = output.max(1)
            ## TODO
            # print("kernel_distance, pred")
            # print(kernel_distance.shape, pred.shape)
            # print(pred[0:6])
            # print(target[0:6])
            ## TODO: ???? why we know labels of OOD samples? What if labels are in different distribution?
            accuracy = pred.eq(target)
            
            accuracies.append(accuracy.cpu().numpy())

            scores.append(-kernel_distance.cpu().numpy())
            # print('score:', scores[0:10])
            ## TODO: hist

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies


def get_auroc_ood(true_dataset, ood_dataset, model, l_gradient_penalty, length_scale, OOD_name):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies[:len(true_dataset)]) #/？？？
    # ood_accuracy = 1 - np.mean(accuracies[len(true_dataset):])
    # print("shape:", anomaly_targets.shape, scores.shape)
    # print("score:", scores[0])
    # print("acc:", accuracy, ood_accuracy)
    score_InD, score_OOD = scores[:len(true_dataset)], scores[len(true_dataset):]
    
    q95= np.percentile(score_InD, 5)
    ood_accuracy = np.count_nonzero(score_OOD < -0.5) / len(score_OOD)
    roc_auc = roc_auc_score(anomaly_targets, scores)
    # plt.figure()
    accuracy = np.count_nonzero(score_InD > -0.5) / len(score_InD)

    # Separating the scores based on anomaly_targets
    scores_normal = scores[anomaly_targets == 0]
    scores_anomalies = scores[anomaly_targets == 1]

    # Plotting two histograms on the same graph with different transparencies
    plt.hist(scores_normal, alpha=0.5, label='Normal', bins=10)
    plt.hist(scores_anomalies, alpha=0.5, label='Anomalies', bins=10)

    # Setting x-axis range from -1 to 0
    plt.xlim(-1, 0)

    # Adding legend
    plt.legend()

    # Adding titles and labels
    plt.title('Histogram of Scores by Target Class')
    plt.xlabel('Scores')
    plt.ylabel('Frequency')

    # Save the figure
    plt.savefig('scatter_plot_'+OOD_name+'_'+str(l_gradient_penalty)+'_'+str(length_scale)+'.png', dpi=300) 


    return accuracy, ood_accuracy, roc_auc


def get_auroc_classification(dataset, model):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)

    return accuracy, roc_auc


## Modify below

# def get_cifar_svhn_ood(model):
#     _, _, _, cifar_test_dataset = get_CIFAR10()
#     _, _, _, svhn_test_dataset = get_SVHN()

#     return get_auroc_ood(cifar_test_dataset, svhn_test_dataset, model)


# def get_fashionmnist_mnist_ood(model):
#     _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
#     _, _, _, mnist_test_dataset = get_MNIST()

#     return get_auroc_ood(fashionmnist_test_dataset, mnist_test_dataset, model)


# def get_fashionmnist_notmnist_ood(model):
#     _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
#     _, _, _, notmnist_test_dataset = get_notMNIST()

#     return get_auroc_ood(fashionmnist_test_dataset, notmnist_test_dataset, model)
