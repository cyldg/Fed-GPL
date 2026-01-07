import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from scipy.special import softmax
import numpy as np
def communication_fedvpt(server_model, models, client_weights, client_num):
    with torch.no_grad():
        # server prompt
        server_prompt_state_dict = server_model.obtain_prompt() # head, prompt_tokens

        # head
        for key in server_prompt_state_dict["head"]:  # weight, bias
            temp = torch.zeros_like(server_prompt_state_dict["head"][key], dtype=torch.float32)
            for client_idx in range(client_num):
                client_prompt_state_dict = models[client_idx].obtain_prompt()
                temp += client_weights[client_idx] * client_prompt_state_dict["head"][key]
            server_prompt_state_dict["head"][key].data.copy_(temp)

        # Prompt_Tokens
        temp1 = torch.zeros_like(server_prompt_state_dict["Prompt_Tokens"], dtype=torch.float32)          
        for client_idx in range(client_num):
            client_prompt_state_dict = models[client_idx].obtain_prompt()
            temp1 += client_weights[client_idx] * client_prompt_state_dict["Prompt_Tokens"]
        server_prompt_state_dict["Prompt_Tokens"].data.copy_(temp1)

        # aggregate and distribute
        server_model.load_prompt(server_prompt_state_dict)
        for client_idx in range(client_num):
            models[client_idx].load_prompt(server_prompt_state_dict)

def communication_fedvpt_device(server_model, models, client_num):
    with torch.no_grad():
        # server prompt
        server_prompt_state_dict = server_model.obtain_prompt() # head, prompt_tokens
        weight = float(0.2)
        for i in range(6):

            # head
            for key in server_prompt_state_dict["head"]:  # weight, bias
                temp = torch.zeros_like(server_prompt_state_dict["head"][key], dtype=torch.float32)
                for client_idx in range(i*5,i*5+5):
                    client_prompt_state_dict = models[client_idx].obtain_prompt()
                    temp += weight * client_prompt_state_dict["head"][key]
                server_prompt_state_dict["head"][key].data.copy_(temp)

            # Prompt_Tokens
            temp1 = torch.zeros_like(server_prompt_state_dict["Prompt_Tokens"], dtype=torch.float32)          
            for client_idx in range(i*5,i*5+5):
                client_prompt_state_dict = models[client_idx].obtain_prompt()
                temp1 += weight * client_prompt_state_dict["Prompt_Tokens"]
            server_prompt_state_dict["Prompt_Tokens"].data.copy_(temp1)

            # aggregate and distribute
            server_model.load_prompt(server_prompt_state_dict)
            for client_idx in range(i*5,i*5+5):
                models[client_idx].load_prompt(server_prompt_state_dict)



def communication_fedavg(server_model,models):
    with torch.no_grad():

        # 初始化全局模型的所有参数
        global_params = [torch.zeros_like(param) for param in models[0].parameters()]
        # print(global_params)
        # print(len(global_params))

        # 对每个客户端模型的所有参数进行加权平均
        for client_model in models:
            for global_param, client_param in zip(global_params, client_model.parameters()):
                global_param.data += client_param.data / len(models)

        # 将聚合后的参数设置到全局模型中
        for global_param, global_model_param in zip(global_params, server_model.parameters()):
            global_model_param.data = global_param.data
        for client_model in models:
            for global_param, model_param in zip(global_params, client_model.parameters()):
                model_param.data = global_param.data

def communication_fedbn(server_model,models):
    with torch.no_grad():
        # 初始化全局模型的所有参数，BN层除外
        global_params = [torch.zeros_like(param) for param in models[0].parameters()]
        bn_layers = ['norm', 'bn']  # 假设Batch Normalization层名字包含'norm'或'bn'

        # 对每个客户端模型的非BN层参数进行加权平均
        for client_model in models:
            for (global_param, (name, client_param)) in zip(global_params, client_model.named_parameters()):
                if not any(bn in name for bn in bn_layers):  # 排除BN层
                    global_param.data += client_param.data / len(models)

        # 将聚合后的参数设置到全局模型中
        for (global_param, (name, global_model_param)) in zip(global_params, server_model.named_parameters()):
            if not any(bn in name for bn in bn_layers):  # 排除BN层
                global_model_param.data = global_param.data

        # 更新客户端模型的非BN层参数
        for client_model in models:
            for (global_param, (name, model_param)) in zip(global_params, client_model.named_parameters()):
                if not any(bn in name for bn in bn_layers):  # 排除BN层
                    model_param.data = global_param.data
                    # print(name)



def communication_PGVIT(server_model,models):
    with torch.no_grad():

        # 初始化全局模型的所有参数
        global_params = [torch.zeros_like(param) for param in models[0].parameters()]
        # print(global_params)
        # print(len(global_params))

        # 对每个客户端模型的所有参数进行加权平均
        for client_model in models:
            for global_param, client_param in zip(global_params, client_model.parameters()):
                global_param.data += client_param.data / len(models)

        # 将聚合后的参数设置到全局模型中
        for global_param, global_model_param in zip(global_params, server_model.parameters()):
            global_model_param.data = global_param.data
        for client_model in models:
            for global_param, model_param in zip(global_params, client_model.parameters()):
                model_param.data = global_param.data

    








# vpt client train
def vpt_train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    model.to(device)
    loss_all = 0
    total = 0
    correct = 0
    i =  1
    for data, target in data_loader:

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
        # print(f'dataloader:{i}/{len(data_loader)}')
        i+=1
        # print(len(data_loader))
        # print(total)

    return loss_all / len(data_loader), correct/total

# # vpt client train ISIC
# def ISIC_vpt_train(model, data_loader, optimizer, loss_fun, device):
#     model.train()
#     model.to(device)
#     loss_all = 0
#     total = 0
#     correct = 0
#     y_true = []
#     y_predict = []
#     y_pred = []
#     for data, target in data_loader:

#         optimizer.zero_grad()
#         data = data.to(device)
#         target = target.to(device)
#         output = model(data)
#         loss = loss_fun(output, target)

#         loss_all += loss.item()
#         total += target.size(0)
#         pred = output.data.max(1)[1]
#         # print(pred)
#         correct += pred.eq(target.view(-1)).sum().item()

#         target = target.cpu().numpy()
#         pred = pred.cpu().numpy()
#         output = output.cpu()

#         y_pred.append(pred)
#         y_true.extend(target)
#         y_predict.extend(softmax(output.detach().numpy(),axis = 1))
#         # print(softmax(output.detach().numpy(),axis = 1),target)
#         # print(correct/total)
#         loss.backward()
#         optimizer.step()
#     y_predict = np.array(y_predict)
#     score = roc_auc_score(y_true,y_predict[:, 1])
    

#     return loss_all / len(data_loader), correct/total,score

# vpt client train ISIC
def ISIC_vpt_train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    model.to(device)
    loss_all = 0
    total = 0
    correct = 0
    y_true = []
    y_predict = []
    y_pred = []
    for data, target in data_loader:

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        # print(pred)
        correct += pred.eq(target.view(-1)).sum().item()

        target = target.cpu().numpy()
        pred = pred.cpu().numpy()
        output = output.cpu()

        y_pred.extend(pred)
        output = output.cpu()
        y_true.extend(target)
        y_predict.extend(softmax(output.detach().numpy(),axis = 1))
    y_predict = np.array(y_predict)
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_predict = np.array(y_predict)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_predict[:, 1])
    average_precision = average_precision_score(y_true, y_predict[:, 1])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    score = roc_auc_score(y_true,y_predict[:, 1])
    
    return loss_all / len(data_loader), correct/total,score
    # return loss_all / len(data_loader), correct/total,score,accuracy,precision,recall,f1,roc_auc,average_precision,specificity

def ISIC_vpt_prox_train(model,server_model, data_loader, optimizer, loss_fun, device, criterion):
    model.train()
    model.to(device)
    loss_all = 0
    total = 0
    correct = 0
    y_true = []
    y_predict = []
    y_pred = []
    for data, target in data_loader:

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # loss = loss_fun(output, target)
        loss = criterion(model,server_model,output,target)
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        # print(pred)
        correct += pred.eq(target.view(-1)).sum().item()

        target = target.cpu().numpy()
        pred = pred.cpu().numpy()
        output = output.cpu()

        y_pred.extend(pred)
        output = output.cpu()
        y_true.extend(target)
        y_predict.extend(softmax(output.detach().numpy(),axis = 1))
    y_predict = np.array(y_predict)
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_predict = np.array(y_predict)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_predict[:, 1])
    average_precision = average_precision_score(y_true, y_predict[:, 1])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    score = roc_auc_score(y_true,y_predict[:, 1])
    

    return loss_all / len(data_loader), correct/total,score,accuracy,precision,recall,f1,roc_auc,average_precision,specificity

# vpt client train prox

def vpt_prox_train(model,server_model, data_loader, optimizer, loss_fun, device,criterion):
    model.train()
    server_model.to(device)
    loss_all = 0
    total = 0
    correct = 0

    for data, target in data_loader:

        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        # loss = loss_fun(output, target)
        loss = criterion(model,server_model,output,target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()



    return loss_all / len(data_loader), correct/total

# mix
# vpt client train

def mix_vpt_train(model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    i =  0
    for batch_idx, (batch1, batch2, batch3) in enumerate(data_loader):
        x1, y1 = batch1
        x2, y2 = batch2
        x3, y3 = batch3
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        y3 = y3.to(device)
        #  solve Nan problem
        if torch.isnan(x1).any() or torch.isinf(x1).any():
            x1, y1 = batch3
            x1 = x1.to(device)
            y1 = y1.to(device)

        if torch.isnan(x2).any() or torch.isinf(x2).any():
            x2, y2 = batch3
            x2 = x2.to(device)
            y2 = y2.to(device)
            
        # 将三个 batch 数据拼接在一起
        data = torch.cat((x1, x2, x3), dim=0)
        target = torch.cat((y1, y2, y3), dim=0)
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        if torch.isnan(data).any() or torch.isinf(data).any():
            raise ValueError("Data contains NaN or inf")

        output = model(data)
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     raise ValueError("Model output contains NaN or inf")
        loss = loss_fun(output, target)
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     raise ValueError("Loss is NaN or inf")
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
        i+=1


    return loss_all /i, correct/total


def mixed_vpt_train(model, mixed_dataloader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    i =  1

    for batch_idx, (x1, x2,x3, y1, y2,y3) in enumerate(mixed_dataloader):

        x1, x2, x3, y1, y2,y3 = x1.to(device), x2.to(device), x3.to(device), \
                                y1.to(device), y2.to(device), y3.to(device)

        x = torch.cat((x1, x2,x3))
        y = torch.cat((y1,y2,y3))


        optimizer.zero_grad()

        output = model(x)
        loss = loss_fun(output, y)

        loss_all += loss.item()
        total += y.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()

        loss.backward()
        optimizer.step()
        # print(f'dataloader:{i}/{len(data_loader)}')
        i+=1
        # print(len(data_loader))
        # print(total)

    return loss_all / len(data_loader), correct/total


# Benign Test vpt
def test_vpt(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0

    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device) 
        output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct/total



def test_vpt_probablity(model, data_loader,device):
    model.eval()
    model = model.to(device)
    for data, target in data_loader:
        data = data.to(device)
        print(data.shape)
        target = target.to(device) 
        output = model(data)
        pred = output.data.max(1)[1]
        print(pred,target,output)
  


def test_ISIC_vpt(model, data_loader, loss_fun, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    y_true = []
    y_predict = []
    y_pred = []
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device) 
        output = model(data)
        loss = loss_fun(output, target)

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]

        
        correct += pred.eq(target.view(-1)).sum().item()
        target = target.cpu().numpy()
        pred = pred.cpu().numpy()

        y_pred.extend(pred)
        output = output.cpu()
        y_true.extend(target)
        y_predict.extend(softmax(output.detach().numpy(),axis = 1))
    y_predict = np.array(y_predict)
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_predict = np.array(y_predict)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_predict[:, 1])
    average_precision = average_precision_score(y_true, y_predict[:, 1])

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    score = roc_auc_score(y_true,y_predict[:, 1])
    

    return loss_all / len(data_loader), correct/total,score,accuracy,precision,recall,f1,roc_auc,average_precision,specificity


#pure test
def test_(model, data_loader,  device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:

        data = data.to(device)
        target = target.to(device) 
        output = model(data)


        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return  correct/total

def test_single(model, data_loader, device):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device) 
            output = model(data)

            total += target.size(0)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    return correct / total
