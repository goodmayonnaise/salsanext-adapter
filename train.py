import torch 
from tqdm import tqdm
from time import time

from mmseg.core import eval_metrics
from mmseg.core.evaluation.metrics import mean_iou

from utils.logs import AverageMeter, ProgressMeter
from utils.metrics import iou, pixel_acc2

def train(model, epochs, train_loader, val_loader, optimizer, criterion, nclasses, scheduler,
          model_path, earlystop, device, metrics, writer_train, writer_val):
    
    since = time()
    best_miou = 0.0

    for epoch in range(epochs) :
        print(f'\n-----------------------\nEpoch {epoch+1}')
        b_t = AverageMeter('train time', ':6.3f')
        d_t = AverageMeter('train data time', ':6.3f')
        loss_running = AverageMeter('train loss', ':.4f')
        miou_running = AverageMeter('train mIoU', ':.4f')
        p_acc_running = AverageMeter('train pixel acc', ':.4f')
        progress = ProgressMeter(len(train_loader),
                                 [b_t, d_t, loss_running, miou_running, 
                                  p_acc_running],
                                  prefix=f"epoch {epoch+1} Train")

        model.train()
        end = time()
        
        with torch.set_grad_enabled(True):
            for iter, batch in enumerate(train_loader):
                d_t.update(time()-end)
                optimizer.zero_grad()

                inputs = batch[0].to(device) # 3 256 1024
                labels = batch[1].to(device) # 3 256 1024
                targets = batch[2].to(device) # 20 256 1024

                outputs = model(inputs)
                loss = criterion(outputs, targets) 
                loss.backward()
                optimizer.step()

                # statistics
                bs = inputs.size(0)
                loss = loss.item()
                loss_running.update(loss, bs)
                progress.display(iter)
                b_t.update(time()-end)
                end = time()
            
                miou = iou(outputs, targets, nclasses, ignore_class=0) 
                p_acc = pixel_acc2(outputs, targets, ignore_class=0)
                miou_running.update(miou)
                p_acc_running.update(p_acc)

                # del batch
                # torch.cuda.empty_cache()

        train_loss = loss_running.avg
        train_miou = miou_running.avg
        train_acc = p_acc_running.avg

        metrics['train_loss'].append(train_loss)
        metrics['train_miou'].append(train_miou)
        metrics['train_acc'].append(train_acc)

        print('\ntrain loss {:.4f} | train miou {:.4f} | train acc {:.4f}'.format(train_loss, train_miou, train_acc))

        val_loss, val_miou, val_acc = val(model, criterion, epoch, val_loader, nclasses, device)
        metrics['val_loss'].append(val_loss)
        metrics['val_miou'].append(val_miou)
        metrics['val_acc'].append(val_acc)

        scheduler.step()
        
        # save history
        with open(f'./{model_path}/result.csv', 'a') as epoch_log:
            epoch_log.write('{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(epoch, train_loss, val_loss, train_miou, val_miou, train_acc, val_acc))


        # save model per epochs         --------------------------------------------------
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(), # Encoder Decoder 따로 저장을 고려할 때 더 자세히 파보면 가능할 것 같다. (전지연)
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'last_val_miou': val_miou,
                    'metrics': metrics,
                    }, f'./{model_path}/last_weights.pth.tar')

        # Save best miou model to file       --------------------------------------------------
        if val_miou > best_miou:
            print('mIoU improved from {:.4f} to {:.4f}.'.format(best_miou, val_miou))
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_miou': best_miou,
                'metrics': metrics}, f'./{model_path}/best_miou_weights.pth.tar')
        
        # early stopping                --------------------------------------------------
        earlystop(val_loss=val_loss, model=model, epoch=epoch, optimizer=optimizer, best_miou=best_miou, metrics=metrics)
        if earlystop.early_stop:
            break

        # tensorboard                   --------------------------------------------------
        writer_train.add_scalar("Loss", train_loss, epoch)
        writer_train.add_scalar("mIoU", train_miou, epoch)
        writer_train.add_scalar("pixel_acc", train_acc, epoch)
        writer_val.add_scalar("Loss", val_loss, epoch)
        writer_val.add_scalar("mIoU", val_miou, epoch)
        writer_val.add_scalar("pixel_acc", val_acc, epoch)
        writer_train.flush()
        writer_train.close()
        writer_val.flush()
        writer_val.close()

        time_elapsed = time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))




def val(model, criterion, epoch, val_loader, nclasses, device):

    b_t = AverageMeter('val time', ':6.3f')
    d_t = AverageMeter('val data time', ':6.3f')
    loss_running = AverageMeter('val Loss', ':.4f')
    miou_running = AverageMeter('val mIoU', ':.4f')
    p_acc_running = AverageMeter('val pixel acc', ':.4f')
    progress = ProgressMeter(len(val_loader),
                             [b_t, d_t, loss_running, miou_running, p_acc_running],
                             prefix=f'epoch {epoch+1} Test')
    
    model.eval()
    with torch.no_grad():
        end = time()
        for iter, batch in enumerate(val_loader):
            d_t.update(time()-end)

            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            targets = batch[2].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            bs = inputs.size(0)
            loss = loss.item()
            loss_running.update(loss, bs)
            b_t.update(time()-end)
            end = time()
            progress.display(iter)

            miou = iou(outputs, targets, nclasses, ignore_class=0) 
            p_acc = pixel_acc2(outputs, targets, ignore_class=0)
            miou_running.update(miou)
            p_acc_running.update(p_acc)
            # del batch
            # torch.cuda.empty_cache()

    val_loss = loss_running.avg
    val_miou = miou_running.avg
    val_acc = p_acc_running.avg
    print('\nvalidation loss {:.4f} | validation miou {:.4f} | validation acc {:.4f}'.format(val_loss, val_miou, val_acc))

    return val_loss, val_miou, val_acc
