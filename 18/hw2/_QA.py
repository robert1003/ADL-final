import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm

class QAExample:

    def __init__(self, title, qas_id, question_text, context_text, answer_text, start_position_character, answerable, answers):
        self.title = title
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.start_position_character = start_position_character
        self.answerable = answerable
        self.answers = answers


    def __str__(self):
        return '( \
                \n\t title: {} \
                \n\t qas_id: {} \
                \n\t question_text: {} \
                \n\t context_text: {} \
                \n\t answer_text: {} \
                \n\t start_position_character: {} \
                \n\t answerable: {} \
                \n\t answers: {} \
                \n)'.format(self.title, self.qas_id, self.question_text, self.context_text, self.answer_text, self.start_position_character, self.answerable, self.answers)

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

def QAExample_to_QAFeature(QAExample, tokenizer, set_type, max_length=512, max_question_length=64, pad_to_max_length=True):
    context_text = tokenizer.tokenize(QAExample.context_text)
    question_text = tokenizer.tokenize(QAExample.question_text)
    if set_type == 'train':
        answer_text = tokenizer.tokenize(QAExample.answer_text)
        start_position = len(tokenizer.tokenize(QAExample.context_text[:QAExample.context_text.find(QAExample.answer_text)]))
        end_position = start_position + len(answer_text)

        if len(context_text) + len(question_text) + 3 > max_length:
            if len(question_text) > max_question_length:
                question_text = question_text[:max_question_length]
            left = max_length - 3 - len(question_text)
            if end_position > left:
                context_text = context_text[end_position - left:end_position]
                end_position = left
                start_position = end_position - len(answer_text)
            else:
                context_text = context_text[:left]
    offset = len(question_text) + 1
    context_text = tokenizer.convert_tokens_to_string(context_text).replace('#', '')
    question_text = tokenizer.convert_tokens_to_string(question_text).replace('#', '')

    dic = tokenizer.encode_plus(
        question_text, context_text, max_length=max_length, pad_to_max_length=pad_to_max_length, truncation_strategy='longest_first'
    )

    input_ids = dic['input_ids']
    attention_mask = dic['attention_mask']
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
