import json
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from _QA import QAExample, QAFeature

def _create_examples(input_data, set_type):
    examples = []
    for entry in tqdm(input_data, desc='[*] Creating'):
        title = entry['title']
        for paragraph in entry['paragraphs']:
            context_text = paragraph['context']
            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                answer_text = None
                start_position_character = None
                answerable = None
                answers = []

                if set_type == 'train':
                    answer = qa['answers'][0]
                    answer_text = answer['text']
                    answerable = qa['answerable']
                    start_position_character = answer['answer_start']
                elif set_type == 'dev':
                    answers = qa['answers']
                    answerable = qa['answerable']

                example = QAExample(
                    title=title,
                    qas_id=qas_id,
                    question_text=question_text,
                    context_text=context_text,
                    answer_text=answer_text,
                    start_position_character=start_position_character,
                    answerable=answerable,
                    answers=answers
                )

                examples.append(example)
    return examples

def read_train_examples(fname):
    with open(fname, 'r') as f:
        s = json.loads(f.readline().strip())
    return _create_examples(s['data'], 'train')

def read_dev_examples(fname):
    with open(fname, 'r') as f:
        s = json.loads(f.readline().strip())
    return _create_examples(s['data'], 'dev')

def read_test_examples(fname):
    with open(fname, 'r') as f:
        s = json.loads(f.readline().strip())
    return _create_examples(s['data'], 'test')

def predict(model, dataset, batch_size, device):
    model.eval()
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    starts = []
    ends = []
    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            start_logits, end_logits = model(**inputs)

            starts.append(start_logits)
            ends.append(end_logits)
            print('[*] Predicting... {}/{}'.format(step + 1, len(test_loader)), end='\r')
        print('')

    return torch.cat(starts, dim=0), torch.cat(ends, dim=0)

def post_process(model, examples, features, dataset, batch_size, tokenizer, threshold, max_len, device):
    starts, ends = predict(model, dataset, batch_size, device)
    results = []
    for step, (example, feature, start_logits, end_logits) in enumerate(zip(examples, features, starts, ends)):
        start_logits = torch.softmax(start_logits, dim=0)
        end_logits = torch.softmax(end_logits, dim=0)

        start_scores, start_ids = start_logits.topk(2, dim=0)
        end_scores, end_ids = end_logits.topk(2, dim=0)

        text_start_idx = feature.tokens.index('[SEP]') + 1
        text_end_idx = feature.tokens.index('[SEP]', text_start_idx) - 1

        possible = []
        for start_score, start_idx in zip(start_scores, start_ids):
            for end_score, end_idx in zip(end_scores, end_ids):
                if start_idx < text_start_idx or start_idx > end_idx or end_idx - start_idx > max_len:
                    continue
                else:
                    possible.append((
                        start_score + end_score,
                        tokenizer.decode(feature.input_ids[start_idx:end_idx + 1], skip_special_tokens=True).replace(' ', '')
                    ))
        possible = sorted(possible, key=lambda t: t[0], reverse=True)

        null_score = ((start_scores[0] + end_scores[0]) * 0.5).detach().cpu().numpy()
        if len(possible) == 0 or null_score < threshold:
            predicted = ""
        else:
            predicted = possible[0][1]

        results.append((example.qas_id, predicted))
        print('[*] Processing... {}/{}'.format(step + 1, len(examples)), end='\r')
    print('')

    return results

def write_prediction(results, fname):
    ans = {}
    for idx, answer in results:
        ans[idx] = answer

    with open(fname, 'w') as f:
        ans = json.dumps(ans)
        print(ans, file=f)

def train(model, dataset, batch_size, 
          learning_rate, adam_epsilon,
          epochs, gradient_accumulation_steps,  
          max_grad_norm, 
          model_fname, helper_fname,
          device):
    # get time and train loader
    start_time = time.time()
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs / gradient_accumulation_steps
    ) 

    model.zero_grad()
    global_step = 0
    # start training!
    for epoch in range(epochs):
        tot_loss = 0
        for step, batch in enumerate(train_loader):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'start_positions': batch[3],
                'end_positions': batch[4],
            }
            outputs = model(**inputs)
            loss = outputs[0] / gradient_accumulation_steps
            loss.backward()
            tot_loss += loss.item()
            
            print('epoch {}/{}: {}/{} loss={}'.format(epoch + 1, epochs, step + 1, len(train_loader), loss.item()), end='\r')

            # update model
            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

        # save bert only
        torch.save({
            'bert_state_dict': model.bert.state_dict()
        }, model_fname)
        '''
        model.save_pretrained(model_fname)
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'schedule_state_dict': scheduler.state_dict()
        }, helper_fname)
        '''

        # print info
        print('                                                          ', end='\r')
        print('epoch {}/{}: total loss={}'.format(epoch + 1, epochs, tot_loss * gradient_accumulation_steps / len(train_loader)))
        print('time passed:', (time.time() - start_time) / 60)
