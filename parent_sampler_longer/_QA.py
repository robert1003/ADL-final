import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm

class QAExample:

    def __init__(self, document_id, index, page_id, context_text, question_text, answer_text):
        self.document_id = document_id
        self.index = index
        self.page_id = page_id
        self.context_text = context_text
        self.question_text = question_text
        self.answer_text = answer_text
        self.answerable = (answer_text != '')

    def __str__(self):
        return '( \
                \n\t document_id: {} \
                \n\t index: {} \
                \n\t page_id: {} \
                \n\t question_text: {} \
                \n\t context_text: {} \
                \n\t answer_text: {} \
                \n\t answerable: {} \
                \n)'.format(self.document_id, self.index, self.page_id, self.question_text, self.context_text, self.answer_text, self.answerable)

class QAFeature:

    def __init__(self, example_id, input_ids, attention_mask, token_type_ids, start_position, end_position, tokens):
        self.example_id = example_id
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.tokens = tokens

    def __str__(self):
        return '(\
                \n\t example_id: {} \
                \n\t input_ids: {} \
                \n\t attention_mask: {} \
                \n\t token_type_ids: {} \
                \n\t start_position: {} \
                \n\t end_position: {} \
                \n\t tokens: {} \
                \n)'.format(self.example_id, self.input_ids, self.attention_mask, self.token_type_ids, self.start_position, self.end_position, self.tokens)
    
def QAExample_to_QAFeature(idx, QAExample, tokenizer, set_type, max_length=512, max_question_length=14, pad_to_max_length=True):
    context_text = tokenizer.tokenize(QAExample.context_text)
    question_text = tokenizer.tokenize(QAExample.question_text)
    assert len(question_text) <= max_question_length
    assert len(question_text) + len(context_text) + 3 <= 512

    if len(question_text) < max_question_length:
        question_text += ['[PAD]'] * (max_question_length - len(question_text))
    
    positions = []
    offset = 0#len(question_text) + 1
    if set_type == 'train' and QAExample.answerable:
        ans_t = QAExample.answer_text
        pos = QAExample.context_text.find(ans_t)
        if pos == -1:
            for ttext in QAExample.answer_text.split('[SEP]'):
                for text in ttext.split(';'):
                    pos = QAExample.context_text.find(text)
                    if pos == -1:
                        print(text, QAExample)
                        continue
                    start_pos = len(tokenizer.tokenize(QAExample.context_text[:pos]))
                    end_pos = start_pos + len(tokenizer.tokenize(text))
                    positions.append((start_pos + offset, end_pos + offset))
        else:
            start_pos = len(tokenizer.tokenize(QAExample.context_text[:pos]))
            end_pos = start_pos + len(tokenizer.tokenize(ans_t))
            positions.append((start_pos + offset, end_pos + offset))

    context_text = tokenizer.convert_tokens_to_string(context_text).replace('#', '')
    question_text = tokenizer.convert_tokens_to_string(question_text).replace('#', '')

    dic = tokenizer.encode_plus(
        question_text, context_text, max_length=max_length, pad_to_max_length=pad_to_max_length, truncation_strategy='do_not_truncate'
    )

    input_ids = dic['input_ids']
    attention_mask = np.array(dic['attention_mask'])
    attention_mask[np.array(input_ids) == 0] = 0
    attention_mask = list(attention_mask)
    try:
        token_type_ids = dic['token_type_ids']
    except:
        token_type_ids = [0] * len(input_ids)

    if set_type == 'train':
        if QAExample.answerable:
            for i, pos in enumerate(positions):
                positions[i] = (pos[0] + 1, pos[1])
        else:
            positions = [(0, 0)]
    else:
        positions = [(None, None)]
        
    return [QAFeature(
        example_id=idx,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        start_position=start_pos,
        end_position=end_pos,
        tokens=tokenizer.tokenize('[CLS]' + question_text + '[SEP]' + context_text + '[SEP]')
    ) for start_pos, end_pos in positions]

def QAExamples_to_QAFeatureDataset(examples, tokenizer, set_type):
    features = []
    for i, example in tqdm(enumerate(examples), desc='[*] Converting'):
        features += QAExample_to_QAFeature(i, example, tokenizer, set_type)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    if set_type == 'train':
        all_start_position = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_position = torch.tensor([f.end_position for f in features], dtype=torch.long)

        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids,
            all_start_position,
            all_end_position,
        )
    else:
        dataset = TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_token_type_ids
        )

    return features, dataset
