import os
import sys
import logging
import torch
import time
from _utils import timeSince, gen_name, score

def Train(model, train_dataloader, dev_dataloader, dev_index_list, criterion, optimizer, device, model_file, dev_ref_file, epochs=10):
    model = model.to(device)

    best_f1 = 0
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
        raw_start_logits = []
        raw_end_logits = []
        with torch.no_grad():
            for (input_ids, attention_mask, token_type_ids, start_position, end_position) in dev_dataloader:
                step += 1

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                start_position = start_position.to(device)
                end_position = end_position.to(device)

                output = model(input_ids, attention_mask, token_type_ids)
                raw_start_logits.append(output[0].cpu())
                raw_end_logits.append(output[1].cpu())

                cur_loss = (criterion(output[0], start_position) + criterion(output[1], end_position)) / 2
                dev_loss += cur_loss.item()

                print('Epoch {}/{} Iter {}/{} Cur_loss {:.5f}'.format(epoch + 1, epochs, step, len(dev_dataloader), cur_loss.item()), end='\r')
        raw_start_logits = torch.cat(raw_start_logits, dim=0)
        raw_end_logits = torch.cat(raw_end_logits, dim=0)

        # calculate f1 score
        prediction = post_process(raw_start_logits, raw_end_logits, QAExamples, QAFeatures, tokenizer)
        name = gen_name() + '.csv'
        Output(prediction, dev_index_list, name)
        cur_f1 = score(dev_ref_file, name)
        os.system(f'rm {name}')

        dev_loss /= step
        logging.info('Epoch {}/{}: training loss = {:.5f}, dev loss = {:.5f}, dev f1 score = {:.5f}, Time {}'.format(
            epoch + 1, epochs, train_loss, dev_loss, cur_f1, timeSince(start, epoch + 1, epochs))) 

        if best_f1 < cur_f1:
            logging.info('Update best loss: {:.5f} -> {:.5f}, saving model to {}'.format(best_loss, dev_loss, model_file))
            best_f1 = cur_f1
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'opt_dict': optimizer.state_dict(),
                    'best_dev_f1': best_f1,
                    'iter': epoch + 1
                },
                model_file
            )

    logging.info('Best dev loss: {:.5f}'.format(best_loss))
