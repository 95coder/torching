import time
import torch
import copy


def train_classifer(model, 
                    criterion, 
                    optimizer, 
                    train_dataloader, 
                    *, 
                    num_epochs=10, 
                    scheduler=None, 
                    device=None, 
                    validation_dataloader=None, 
                    weights_file=None):

    since = time.time()
    
    if validation_dataloader:
        dataloaders = {'train': train_dataloader, 'val': validation_dataloader}
        phases = ['train', 'val']

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

    else:
        dataloaders = {'train': train_dataloader}
        phases = ['train']

    for epoch_i in range(num_epochs):
        print('Epoch {}/{}'.format(epoch_i, num_epochs - 1))
        print('-' * 10)

        for phase in phases:
            if phase == 'train':
                model.train()
            elif phase == 'val':
                model.eval()
        
            running_loss = .0
            running_corrects = 0

            for i, data in enumerate(dataloaders[phase], 0):
                inputs , labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * len(inputs)
                running_corrects += (labels == preds).sum().double()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / (len(dataloaders[phase].dataset))
            
            if phase == 'train':
                scheduler.step()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
        print()
    
    if 'val' in phases:
        print('Best validation Acc: {:.4f}'.format(best_acc))
        model.load_state_dict(best_model_wts)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if weights_file:
        torch.save(model.state_dict(), weights_file)


def test_classifier(model, dataloader, *, device=None):
    num_corrects = 0
    total = 0

    for i, data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        num_corrects += (preds == labels).sum().item()
        total += len(inputs)
    
    print('Test Acc: {:.4f}'.format(num_corrects / total))