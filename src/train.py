import tempfile
import os
import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot, accuracy
from torch.utils.tensorboard import SummaryWriter

device= "cuda" if torch.cuda.is_available()else "cpu"

def to_device(data, label):
    if isinstance(data, torch.Tensor):
        return data.to(device), label.to(device)
        
    data=[d.to(device) for d in data]
    label=label.to(device)
    return data, label
    


def train_one_epoch(train_dataloader, model, optimizer, loss, scaler, accumulation_steps):
    """
    Performs one train_one_epoch epoch
    """

    model.train() # set the model to training mode
    
    train_loss = 0.0
    acc1 = 0
    acc5 = 0
    total = 0
   
    for batch_idx, (data, target) in tqdm(
            enumerate(train_dataloader),
            desc="Training",
            total=len(train_dataloader),
            leave=True,
            ncols=80,
        ):

        # move data to GPU
        data, target = to_device(data, target)

            

        if scaler:
            with torch.autocast(device_type=device, dtype=torch.float16):
                output  = model(data)
                loss_value  = loss(output, target)
            scaler.scale(loss_value).backward()

            if (batch_idx+1) % accumulation_steps ==0 or (batch_idx+1)== len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            output  = model(data)
            loss_value  = loss(output, target)
            loss_value.backward()
            
            if (batch_idx+1) % accumulation_steps ==0 or (batch_idx+1)== len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()


        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )
        nacc1, nacc5=accuracy(output, target, topk=(1,5))
        acc1+=nacc1
        acc5+=nacc5
    return train_loss, (acc1/len(train_dataloader), acc5/len(train_dataloader))


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        valid_loss = 0.0
        total = 0
        acc1 = 0
        acc5 = 0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU
            data, target = to_device(data, target)


            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data)
            # 2. calculate the loss
            loss_value  = loss(output, target)
            # NUM CLASS 1

            # loss_value  = loss(output, target.type(torch.float32).view_as(output))


            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )            
            nacc1, nacc5=accuracy(output, target, topk=(1,5))
            acc1+=nacc1
            acc5+=nacc5
            

    return valid_loss, (acc1/len(valid_dataloader), acc5/len(valid_dataloader))


def optimize(data_loaders, model, optimizer, loss, s_epoch, n_epochs, model_name, step ,accumulation_steps = 1, use_amp=True, interactive_tracking=False, run_logs=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_acc_max = None
    logs = {}

    # define scaler for amp GPUs
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None



    model=model.to(device)
    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a plateau
 
    if run_logs:
        path=f"runs/{model_name}"
        writer =SummaryWriter(path)

        if not os.path.isfile(path):
            writer.add_text('model',str(model))
            writer.add_text('optimizer',str(optimizer))

    for epoch in range(s_epoch, s_epoch+n_epochs):

        train_loss, train_top5 =train_one_epoch(data_loaders["train"], model, optimizer, loss, scaler, accumulation_steps)

        valid_loss, valid_top5 = valid_one_epoch(data_loaders["valid"], model, loss) 



        # Log the losses and the current learning rate

        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["acc"]=train_top5[0].cpu()
            logs["val_acc"]=valid_top5[0].cpu()
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()
        if run_logs:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/valid", valid_loss, epoch)
            writer.add_scalar("Acc/train", train_top5[0], epoch)
            writer.add_scalar("Acc/valid",valid_top5[0], epoch)
            writer.add_scalar("5Acc/train",  train_top5[-1], epoch)
            writer.add_scalar("5Acc/valid", valid_top5[-1], epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'],epoch)

        
        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\tTraining Acc: {:.3f} \tValidation Acc: {:.3f}\tTraining 5Acc: {:.3f} \tValidation 5Acc: {:.3f}".format(
                epoch, train_loss, valid_loss,train_top5[0],valid_top5[0], train_top5[-1], valid_top5[-1]
            )
        )

        if valid_acc_max is None or valid_acc_max <= valid_top5[0]:
            print(f"New max accuracy: {valid_top5[0]:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save({
            'epochs': epoch,
            'model_state_dict':model.state_dict(),
            }, f'checkpoints/best_{model_name}.pt')

            valid_acc_max=valid_top5[0]

        # Update learning rate, i.e., make a step in the learning rate scheduler
        #
        torch.save({
                    'epochs': epoch,
                    'model_state_dict':model.state_dict(),
                    }, f'checkpoints/last_{model_name}.pt')
        step(valid_loss, epoch)

        



    if run_logs:
        writer.flush()
        writer.close()




def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU
            data, target = to_device(data, target)


            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits  = model(data)
            # 2. calculate the loss
            
            loss_value  = loss(logits, target)
            # NUM CLASS 1
            
            # loss_value  = loss(output, target.type(torch.float32).view_as(logits))


            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            pred  = torch.argmax(logits, dim=1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0) 

            # NUM CLASS 1
            
            # pred  = torch.round_(torch.sigmoid(logits).cuda())

            # # compare predictions to true label
            # correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            # total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


