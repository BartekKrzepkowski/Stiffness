import os

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

DOWNLOAD = False


def get_mnist(dataset_path):
    dataset_path = dataset_path if dataset_path is not None else os.environ['MNIST_PATH']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_data = datasets.MNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.MNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_fashion_mnist(dataset_path, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['FASHION_MNIST_PATH']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])
    train_data = datasets.FashionMNIST(dataset_path, train=True, download=DOWNLOAD, transform=transform)
    train_eval_data = train_data
    test_data = datasets.FashionMNIST(dataset_path, train=False, download=DOWNLOAD, transform=transform)
    return train_data, train_eval_data, test_data


def get_cifar10(dataset_path=None, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR10_PATH']
    if proper_normalization:
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.262)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    transform_train_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transform_train_2 if whether_aug else transform_eval
    train_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR10(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR10(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data



def get_cifar100(dataset_path, whether_aug=True, proper_normalization=True):
    dataset_path = dataset_path if dataset_path is not None else os.environ['CIFAR100_PATH']
    if proper_normalization:
        mean, std = (0.5071, 0.4865, 0.4409), (0.267, 0.256, 0.276)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_train = transforms.Compose([
        transforms.TrivialAugmentWide(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.1),
    ])
    transform_train_2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(degrees=0, translate=(1/8, 1/8)),
        transforms.Normalize(mean, std),
    ])
    transform_train = transform_train_2 if whether_aug else transform_eval
    train_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_train)
    train_eval_data = datasets.CIFAR100(dataset_path, train=True, download=DOWNLOAD, transform=transform_eval)
    test_data = datasets.CIFAR100(dataset_path, train=False, download=DOWNLOAD, transform=transform_eval)
    return train_data, train_eval_data, test_data



