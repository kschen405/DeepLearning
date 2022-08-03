from turtle import forward
import numpy as np
import torch
from torch import device, nn, tensor
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from matplotlib import pyplot as plt


def read_bci_data():
    S4b_train = np.load('data\S4b_train.npz')
    X11b_train = np.load('data\X11b_train.npz')
    S4b_test = np.load('data\S4b_test.npz')
    X11b_test = np.load('data\X11b_test.npz')

    train_data = np.concatenate(
        (S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate(
        (S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate(
        (S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate(
        (S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label - 1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)
    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    return tensor(train_data), tensor(train_label), tensor(test_data), tensor(test_label)


def activation(act_type):
    if act_type == "ELU":
        return nn.ELU(alpha=1.0)
    elif act_type == "ReLU":
        return nn.ReLU()
    else:
        return nn.LeakyReLU()


class EEGNet(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.act_type = act_type
        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(
                1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True)
        )
        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1),
                      stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            activation(self.act_type),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(
                1, 1), padding=(0, 7), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            activation(self.act_type),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25)
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=736, out_features=2, bias=True)
        )

    def forward(self, x):
        x = self.firstConv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.view(-1, self.classify[0].in_features)
        x = self.classify(x)
        return x


class DeepConvNet(nn.Module):
    def __init__(self, act_type):
        super().__init__()
        self.act_type = act_type

        self.Conv0 = nn.Conv2d(1, 25, (1, 5))
        self.Convs = nn.ModuleList()
        paras = [25, 25, 50, 100, 200]
        kernel_size = [(2, 1), (1, 5), (1, 5), (1, 5)]
        for i in range(4):
            conv_tmp = nn.Sequential(
                nn.Conv2d(paras[i], paras[i+1], kernel_size[i]),
                nn.BatchNorm2d(paras[i+1], eps=1e-5, momentum=0.1),
                activation(self.act_type),
                nn.MaxPool2d((1, 2)),
                nn.Dropout(p=0.5)
            )
            self.Convs.append(conv_tmp)
            self.classify = nn.Linear(8600, 2)

    def forward(self, x):
        x = self.Conv0(x)
        for Conv in self.Convs:
            x = Conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classify(x)
        return x


def train(model, train_dataloader, test_dataloader, optimizer, criterion, num_epoch=300):
    train_acc_list = []
    test_acc_list = []
    max_test_acc = 0
    max_acc_model = None
    model_name = None
    for epoch in range(num_epoch):
        model.train()
        TP_and_TN = 0
        total_loss = 0
        for x, y in train_dataloader:  # batch
            optimizer.zero_grad()
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.long)
            prob = model(x)
            loss = criterion(prob, y)
            loss.backward()
            optimizer.step()
            pred = torch.argmax(prob, dim=1)
            total_loss += loss.item()
            TP_and_TN += torch.count_nonzero(
                torch.where(pred == y, 1, 0)).item()
        # Calculate accuracy and loss
        train_avg_loss = total_loss / len(train_dataloader.dataset)
        train_avg_acc = TP_and_TN / len(train_dataloader.dataset)
        test_avg_acc, test_avg_loss = test(model, test_dataloader, criterion)

        train_acc_list.append(train_avg_acc)
        test_acc_list.append(test_avg_acc)

        if epoch % 25 == 0:
            print(
                f"Epoch = {epoch}, training loss = {train_avg_loss:.3f}, train accuracy = {train_avg_acc:.3f}, test accuracy = {test_avg_acc:.3f}")

        if test_avg_acc >= max_test_acc:
            max_test_acc = test_avg_acc
            max_acc_model = model.state_dict()
            model_name = f"{model.__class__.__name__}_{model.act_type}_{epoch}_{round(test_acc_list[-1], 3)}_batch16lr0.001.pth"
        # if max_test_acc > 0.88:
        #     break
        #     print(f"MAX Test acc = {max(test_acc_list)}% {round(test_acc_list[-1], 3)}")
        # print(f"Highest test accuracy: {max_test_acc:.3f}")
    return train_acc_list, test_acc_list, max_acc_model, model_name


def test(model, test_dataloader, criterion):
    model.eval()
    TP_and_TN = 0
    total_loss = 0
    for x, y in test_dataloader:  # batch
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.long)
        with torch.no_grad():
            prob = model(x)
            loss = criterion(prob, y)
            pred = torch.argmax(prob, dim=1)
            total_loss += loss.item()
            TP_and_TN += torch.count_nonzero(
                torch.where(pred == y, 1, 0)).item()
    avg_loss = total_loss / len(test_dataloader.dataset)
    avg_acc = TP_and_TN / len(test_dataloader.dataset)
    # print(f"Test loss = {avg_loss}, accuracy = {avg_acc}")
    return avg_acc, avg_loss


def load_data():
    train_data, train_label, test_data, test_label = read_bci_data()  # return tensor
    train_dataloader = DataLoader(TensorDataset(
        train_data, train_label), batch_size=64, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(
        test_data, test_label), batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataloader, test_dataloader = load_data()
    # Set seeds
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set model paras
    net_list = ["EEGNet", "DeepConvNet"]
    act_type_list = ["ReLU", "LeakyReLU", "ELU"]
    model_list = ["models\EGGNet_ReLU.pth", "models\EEGNet_LeakyReLU.pth", "models\EEGNet_ELU.pth",
                  "models\DeepConvNet_ReLU.pth", "models\DeepConvNet_LeakyReLU.pth", "models\DeepConvNet_ELU.pth"]
    num_epoch = 300
    criterion = nn.CrossEntropyLoss()

    demo = 1
    demo_cnt = 0
    # Test different models
    for net in net_list:
        train_acc_activations = []
        test_acc_activations = []
        for act_type in act_type_list:
            if net == "EEGNet":
                model = EEGNet(act_type).to(device)
            else:
                model = DeepConvNet(act_type).to(device)
            # normal or demo mode
            if demo == 0:
                # , weight_decay=0.01)
                optimizer = optim.Adam(model.parameters(), lr=2.0E-04)
                train_acc, test_acc, max_model, model_name = train(
                    model, train_dataloader, test_dataloader, optimizer, criterion, num_epoch=num_epoch)
                train_acc_activations.append(train_acc)
                test_acc_activations.append(test_acc)
                print(
                    f"{net} + {act_type}: Highest test accuracy = {max(test_acc):.3f}")
                torch.save(max_model, model_name)
            else:
                ckpt = torch.load(model_list[demo_cnt])
                model.load_state_dict(ckpt)
                test_acc, test_loss = test(model, test_dataloader, criterion)
                print(
                    f"{model_list[demo_cnt]}: Highest test accuracy = {test_acc}")
                demo_cnt += 1

        if demo == 0:
            # plot results
            plt.title(f"Activation Function Comparison({net})")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy(%)")
            for i in range(3):
                plt.plot(train_acc_activations[i],
                         label=f"{act_type_list[i]}_train")
                plt.plot(test_acc_activations[i],
                         label=f"{act_type_list[i]}_test")
            plt.legend()
            plt.savefig(f"Activation_Comparison_{net}_2")
            plt.show()
