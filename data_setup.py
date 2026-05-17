
from utils_py import *

def setup():
    #Downloading the dataset
    download_dataset()

    #Loading the dataset

    images,labels=load_data()


    #Splitting the dataset into training and validation and testing sets

    train_images, val_images, test_images, train_labels, val_labels, test_labels=split_data(images, labels)
    
    #Getting the data transformations
    train_transform, val_transform=get_transforms()


    #Creating PyTorch Datasets 

    train_dataset=MyDataset(train_images, train_labels, transform=train_transform)
    val_dataset=MyDataset(val_images, val_labels, transform=val_transform)
    test_dataset=MyDataset(test_images, test_labels, transform=val_transform)


    #Creating PyTorch DataLoaders

    train_loader, val_loader, test_loader =get_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=16)
    
    return train_loader, val_loader, test_loader ,train_labels


