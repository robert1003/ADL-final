import torch
import numpy as np
from torch.utils.data import TensorDataset
from tqdm import tqdm

class QAExample:

    def __init__(self, document_id, page_id, context_text, question_text, answer_text):
        self.document_id = document_id
        self.page_id = page_id
        self.context_text = context_text
        self.question_text = question_text
        self.answer_text = answer_text
        self.answerable = (answer_text != '')

    def __str__(self):
        return '( \
                \n\t document_id: {} \
                \n\t page_id: {} \
                \n\t question_text: {} \
                \n\t context_text: {} \
                \n\t answer_text: {} \
                \n\t answerable: {} \
                \n)'.format(self.document_id, self.page_id, self.question_text, self.context_text, self.answer_text, self.answerable)

class QAFeature:

    def __init__(self, input_ids, attention_mask, token_type_ids, start_position, end_position, tokens):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.start_position = start_position
        self.end_position = end_position
        self.tokens = tokens

    def __str__(self):
        return '(\n\t input_ids: {} \n\t attention_mask: {} \n\t token_type_ids: {} \n\t start_position: {} \n\t end_position: {} \n\t tokens: {} \n)' \
                .format(self.input_ids, self.attention_mask, self.token_type_ids, self.start_position, self.end_position, self.tokens)

def QAExample_to_QAFeature(QAExample, tokenizer, set_type, max_length=512, max_question_length=14, pad_to_max_length=True):
    context_text = tokenizer.tokenize(QAExample.context_text)
    question_text = tokenizer.tokenize(QAExample.question_text)
    assert len(question_text) <= max_question_length
    assert len(question_text) + len(context_text) + 3 <= 512

    if len(question_text) < max_question_length:
        question_text += ['[PAD]'] * (max_question_length - len(question_text))

    if set_type == 'train':
        answer_text = tokenizer.tokenize(QAExample.answer_text)
        start_position = len(tokenizer.tokenize(QAExample.context_text[:QAExample.context_text.find(QAExample.answer_text)]))
        end_position = start_position + len(answer_text)

    offset = len(question_text) + 1
    context_text = tokenizer.convert_tokens_to_string(context_text).replace('#', '')
    question_text = tokenizer.convert_tokens_to_string(question_text).replace('#', '')

    dic = tokenizer.encode_plus(
        question_text, context_text, max_length=max_length, pad_to_max_length=pad_to_max_length, truncation_strategy='do_not_truncate'
    )

    input_ids = dic['input_ids']
    attention_mask = np.array(dic['attention_mask'])
    attention_mask[np.array(input_ids) == 0] = 0
    attention_mask = list(attention_mask)
    token_type_ids = dic['token_type_ids']

    if set_type == 'train':
        if QAExample.answerable:
            start_position += 1
        else:
            start_position = end_position = 0
    else:
        start_position = end_position = None

    return QAFeature(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        start_position=start_position + offset if start_position is not None else start_position,
        end_position=end_position + offset if end_position is not None else end_position,
        tokens=tokenizer.tokenize('[CLS]' + question_text + '[SEP]' + context_text + '[SEP]')
    )

def QAExamples_to_QAFeatureDataset(examples, tokenizer, set_type):
    features = []
    for example in tqdm(examples, desc='[*] Converting'):
        features.append(QAExample_to_QAFeature(example, tokenizer, set_type))

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



