import csv
import sys

def Output(output_list, index_list, output_file):
    mp = dict()
    for (filename, idx, question, answer) in output_list:
        filename = filename + '-' + str(idx)
        if filename not in mp:
            mp[filename] = []
        mp[filename].append((question, answer))

    output = []
    for (filename, indexes) in sorted(index_list.items(), key=lambda t: int(t[0])):
        for i in indexes:
            name = filename + '-' + str(i)
            predictions = ''
            if name in mp:
                QA_list = mp[name]
                predictions = ' '.join(['{tag}:{value}'.format(tag=tag.replace(' ', ''), value=value.replace(' ', '')) for tag, value in QA_list])
            else:
                predictions = 'NONE'
            output.append({'ID': name, 'Prediction': predictions})

    with open(output_file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, ['ID', 'Prediction'])
        writer.writeheader()
        writer.writerows(output) 
