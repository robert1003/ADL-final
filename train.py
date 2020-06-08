def Train(model, train_dataloader, dev_dataloader, criterion, optimizer, device, epochs = 500):
    model = model.to(device)

    best_loss = 1e10
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
        print('Epoch {}/{}: training loss = {:.5f}, dev loss = {:.5f}'.format(epoch, epochs, train_loss, dev_loss)) 

        if best_loss > dev_loss:
            best_loss = dev_loss
            checkpoint_path = f'model/model.{epoch + 1}.pt'
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'iter': epoch + 1
                },
                checkpoint_path
            )

