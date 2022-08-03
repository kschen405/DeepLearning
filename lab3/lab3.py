import pandas as pd
import torch
from torch import from_numpy, nn, device, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.autograd.function import once_differentiable
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import kornia as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import time
from datetime import datetime

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values) # (28099, 1), (28099, 1)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(Dataset):
    def __init__(self, root, mode):
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, index):
        PATH = os.path.join(self.root, self.img_name[index] + ".jpeg") # os.path.join

        label = torch.from_numpy(np.array(self.label[index]))
        if self.mode == "train":
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
                # transforms.Normalize((0.4749, 0.4602, 0.4857),(0.2526, 0.2780, 0.2291))

            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize((0.4749, 0.4602, 0.4857),(0.2526, 0.2780, 0.2291))
            ])
        img = Image.open(PATH).convert("RGB")
        img = transform(img)
        return img, label

class ResNet(nn.Module):
    def __init__(self, model_type, pretrained=True):
        super(ResNet, self).__init__()
        if model_type == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        else:
            self.model = models.resnet50(pretrained=pretrained)
        fc_in = self.model.fc.in_features
        self.model.fc = nn.Linear(fc_in, 5)

    def forward(self, x):
        x = self.model(x)
        return x





def train(paras, train_loader, test_loader):
    model = ResNet(paras['model'], pretrained=paras['pretrain']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=paras['lr'], momentum=paras['momentum'], weight_decay=paras['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    train_acc_list = []
    test_acc_list = []
    max_acc = 0
    max_model = None
    for epoch in range(paras['num_epoch']):
        start = time.time()
        model.train()
        TP_and_TN = 0
        total_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            # x = K.color_jitter(x)
            # y = K.color_jitter(y)
            x = x.to(device)
            y = y.to(device)
            y_prob = model(x)
            loss = criterion(y_prob, y)
            loss.backward()
            optimizer.step()
            y_pred = torch.argmax(y_prob, axis=1)
            TP_and_TN += torch.count_nonzero(torch.where(y_pred == y, 1, 0)).item()
            total_loss += loss
        train_avg_acc = TP_and_TN / len(train_loader.dataset)
        train_avg_loss = total_loss / len(train_loader.dataset)
        train_acc_list.append(train_avg_acc)
        # test
        test_avg_acc = test(model, paras, test_loader)
        test_acc_list.append(test_avg_acc)
        print(f"Epoch {epoch}, train loss = {train_avg_loss:.5f} train acc = {train_avg_acc:.3f} test acc = {test_avg_acc:.3f} elapse time = {time.time()-start}")
        # max test acc
        if test_avg_acc > max_acc:
            max_acc = test_avg_acc
            max_model = model.state_dict()
            if paras['pretrain']:
                model_name = f"{paras['model']}_pretrain_{max_acc}"
            else:
                model_name = f"{paras['model']}_wthout_pretrain_{max_acc}"
            torch.save(max_model, model_name)
    return train_acc_list, test_acc_list, max_model, model_name

def test(model, paras, test_loader):
    model.eval()
    TP_and_TN = 0
    for x, y in test_loader:
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            y_prob = model(x)
            y_pred = torch.argmax(y_prob, axis=1)
            TP_and_TN += torch.count_nonzero(torch.where(y_pred == y, 1, 0)).item()
    test_avg_acc = TP_and_TN / len(test_loader.dataset)
    return test_avg_acc

def plot_results(paras, test_without, test_pretrain, train_without, train_pretrain):
    plt.title(f"Result Comparison ({paras['model']})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.plot(test_without, label="Test(w/o pretraining)")
    plt.plot(test_pretrain, label="Test(with pretraining)")
    plt.plot(train_without, label="Train(w/o pretraining)")
    plt.plot(train_pretrain, label="Train(with pretraining)")
    plt.legend()
    plt.savefig(f"{paras['model']}")
    # plt.show()

def plot_confusion(paras, y, y_pred):
    print("Enter plot_confusion")
    print(type(y), type(y_pred))
    print(f"y = \n{y}")
    print(f"y_pred = \n{y_pred}")
    y = y.data.cpu().numpy()
    y_pred = y_pred.data.cpu().numpy()
    disp = ConfusionMatrixDisplay.from_predictions(y_true=y, y_pred=y_pred, cmap="cool",normalize="true")
    disp.plot()
    if paras['pretrain']:
        disp.ax_.set_title(f"Normalized confusion matrix (Pretrained {paras['model']})")
        plt.savefig(f"{paras['model']}_pretrain_confusion.png")
    else:
        disp.ax_.set_title(f"Normalized confusion matrix (W/o pretrained {paras['model']})")
        plt.savefig(f"{paras['model']}_without_confusion.png")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    # Set paras
    paras = {
        'model': None,
        'pretrain': None,
        'num_epoch': 10,
        'batch_size': 16,
        'lr': 1e-3,
        'loss_func': 'CrossEntropy',
        'optimizer': 'SGD',
        'weight_decay': 5e-4,
        'momentum': 0.9
    }

    # Set GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # Set seeds
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    ROOT_DIR = os.path.abspath(os.path.join(os.curdir, "data"))
    # mode = "train" if trainmode else "test"
    train_dataset = RetinopathyLoader(ROOT_DIR, "train")
    test_dataset = RetinopathyLoader(ROOT_DIR, "test")
    train_loader = DataLoader(train_dataset,batch_size=paras['batch_size'], num_workers=1, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=paras['batch_size'], num_workers=1, shuffle=False)

    # ========================================= resnet18 ========================================
    # resnet18, pretrain
    print(f"resnet18, pretrain, start time {datetime.now()}")
    paras['model'] = 'resnet18'
    paras['batch_size'] = 32
    paras['pretrain'] = 1
    train_pretrain, test_pretrain, max_model, model_name = train(paras, train_loader, test_loader)
    torch.save(max_model, model_name)
    # resnet18, w/o
    print("resnet18, w/o")
    paras['pretrain'] = 0
    train_without, test_without, max_model, model_name = train(paras, train_loader, test_loader)
    torch.save(max_model, model_name)
    # plot resnet18
    plot_results(paras, test_without, test_pretrain, train_without, train_pretrain)
    torch.cuda.empty_cache()

    # =========================================== resnet50 ========================================
    # resnet50, pretrain
    print(f"resnet50, pretrain, start time {datetime.now()}")
    paras['model'] = 'resnet50'
    paras['pretrain'] = 1
    paras['batch_size'] = 32
    train_pretrain, test_pretrain, max_model, model_name = train(paras, train_loader, test_loader)
    torch.save(max_model, model_name)
    torch.cuda.empty_cache()
    # resnet50, w/o
    print("resnet50, w/o ")
    paras['pretrain'] = 0
    train_without, test_without, max_model, model_name = train(paras, train_loader, test_loader)
    torch.save(max_model, model_name)
    # plot resnet50
    plot_results(paras, test_without, test_pretrain, train_without, train_pretrain)

    # demo = 0
    # if demo == 1:
    #     paras = {
    #         'model': "resnet50",
    #         'pretrain': 1,
    #         'num_epoch': 10,
    #         'batch_size': 32,
    #         'lr': 1e-3,
    #         'loss_func': 'CrossEntropy',
    #         'optimizer': 'SGD',
    #         'weight_decay': 5e-4,
    #         'momentum': 0.9
    #     }

    #     model = ResNet(paras['model'], pretrained=paras['pretrain']).to(device)
    #     ckpt = torch.load("saved_models/resnet50_pretrain.pth")
    #     model.load_state_dict(ckpt)
    #     print("Successfully loaded")
    #     test(model, paras, test_loader, confused=True)
