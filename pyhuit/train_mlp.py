# train_mlp.py
import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from MyDataSet import MyDataSet
from MLP import MLP
import visdom
from tools import plot_confusion_matrix
import sklearn.metrics as m

batchsz = 8
lr = 0.001# learning rate
epoches = 1500
torch.manual_seed(1234)
file_path = "C:/Users/Tardis/Desktop/huit/svg evaluation huitres.csv"

# 读取数据
train_db = MyDataSet(file_path, mode='train')
val_db = MyDataSet(file_path, mode='val')
test_db = MyDataSet(file_path, mode='test')

train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True)
val_loader = DataLoader(val_db, batch_size=batchsz)
test_loader = DataLoader(test_db, batch_size=batchsz)
viz = visdom.Visdom()


# 计算正确率
def evaluate(model, loader):
    """
    :param model: net model
    :param loader: data set
    :return: ACC
    """
    correct = 0
    total = len(loader.dataset)
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
        correct += torch.eq(pred, y).sum().float().item()
    return correct / total


# 计算分类的各种指标
def test_evaluate(model, loader):
    """
       :param model: net model
       :param loader: data set
       :return: ACC
    """
    y_true = []
    predict = []
    for x, y in loader:
        with torch.no_grad():
            logits = model(x)
            result = logits.argmax(dim=1)
            for i in y.numpy():
                y_true.append(i)
            for j in result.numpy():
                predict.append(j)

    #print(classification_report(y_true, predict))
    #plot_confusion_matrix(confusion_matrix(y_true, predict), classes=range(10), title='confusion matrix')
    #print("confusion matrix")
    #print(confusion_matrix(y_true, predict))
    #print("f1-score:{}.".format(m.f1_score(y_true, predict)))
    print(y_true)
    print(predict)
    return accuracy_score(y_true, predict)


def main():
    model = MLP(7, 3)  # initial net model
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    criterion = nn.CrossEntropyLoss()    # loss function

    best_epoch, best_acc = 0, 0
    viz.line([0], [-1], win='loss', opts=dict(title='loss'))   # visdom print loss graph
    viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))  # visdom print val_acc graph

    for epoch in range(epoches):
        for step, (x, y) in enumerate(train_loader):
            # x:[b,16] ,y[b]
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        viz.line([loss.item()], [epoch], win='loss', update='append')

        if epoch % 5 == 0:

            val_acc = evaluate(model, val_loader)
            # train_acc = evaluate(model,train_loader)
            print("epoch:[{}/{}]. val_acc:{}.".format(epoch, epoches, val_acc))

            # print("train_acc", train_acc)
            viz.line([val_acc], [epoch], win='val_acc', update='append')
            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc

                torch.save(model.state_dict(), 'best.mdl')

    print('best acc:{}. best epoch:{}.'.format(best_acc, best_epoch))
    model.load_state_dict(torch.load('best.mdl'))
    print("loaded from ckpt!")

    test_acc = test_evaluate(model, test_loader)
    print("test_acc:{}".format(test_acc))


if __name__ == '__main__':
    main()
