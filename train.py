def Train(model, dataloader, criterion, optimizer, device, epochs = 500):
    model = model.to(device)
    for epoch in range(epochs):
        loss, step = 0, 0
        for (input_ids, attention_mask, token_type_ids, start_position, end_position) in dataloader:
            step += 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            start_position = start_position.to(device)
            end_position = end_position.to(device)

            output = model(input_ids, attention_mask, token_type_ids)
            
            cur_loss = (criterion(output[0], start_position) + criterion(output[1], end_position)) / 2
            loss += cur_loss.item()

            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()

        loss /= step
        print('epoch {}: loss = {:.3f}'.format(epoch, loss)) 
