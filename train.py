import sys
import logging
import torch
import time
from _utils import timeSince

def Train(model, train_dataloader, dev_dataloader, criterion, optimizer, device, model_file, epochs=10):
    model = model.to(device)

    best_loss = 1e10
    start = time.time()
    for epoch in range(epochs): 
        train_loss, step = 0, 0
        model.train()
        for (input_ids, attention_mask, token_type_ids, start_position, end_position) in train_dataloader:
            step += 1

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            start_position = start_position.to(device)
            end_position = end_position.to(device)

            output = model(input_ids, attention_mask, token_type_ids)
            
            cur_loss = (criterion(output[0], start_position) + criterion(output[1], end_position)) / 2
            train_loss += cur_loss.item()

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

            print('Epoch {}/{} Iter {}/{} Cur_loss {:.5f}'.format(epoch + 1, epochs, step, len(train_dataloader), cur_loss.item()), end='\r')

        train_loss /= step

        dev_loss, step = 0, 0
        model.eval()
        with torch.no_grad():
            for (input_ids, attention_mask, token_type_ids, start_position, end_position) in dev_dataloader:
                step += 1

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                start_position = start_position.to(device)
                end_position = end_position.to(device)

                output = model(input_ids, attention_mask, token_type_ids)

                cur_loss = (criterion(output[0], start_position) + criterion(output[1], end_position)) / 2
                dev_loss += cur_loss.item()

                print('Epoch {}/{} Iter {}/{} Cur_loss {:.5f}'.format(epoch + 1, epochs, step, len(dev_dataloader), cur_loss.item()), end='\r')

        dev_loss /= step
        logging.info('Epoch {}/{}: training loss = {:.5f}, dev loss = {:.5f} Time Remaining: {}'.format(
            epoch + 1, epochs, train_loss, dev_loss, timeSince(start, epoch + 1, epochs))) 

        if best_loss > dev_loss:
            print('Update best loss: {:.5f} -> {:.5f}, saving model to {}'.format(best_loss, dev_loss, model_file))
            best_loss = dev_loss
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                    'best_loss': best_loss,
                    'iter': epoch + 1
                },
                model_file
            )

    torch.save(
        {
            'state_dict': model.state_dict(),
            'opt_dict': optimizer.state_dict(),
            'best_loss': best_loss,
            'iter': epochs
        },
        f'{epochs}_model_file'
    )
