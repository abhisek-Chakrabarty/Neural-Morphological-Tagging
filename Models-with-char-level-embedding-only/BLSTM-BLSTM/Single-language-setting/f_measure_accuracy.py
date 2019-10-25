def f_measure_acc_for_single_word(gold, predicted):
    if gold.find('POS=') == -1:
        gold = 'POS=' + gold
    if predicted.find('POS=') == -1:
        predicted = 'POS=' + predicted

    key_value_pair_array_gold = gold.split('|')
    key_value_pair_array_predicted = predicted.split('|')

    gold_dictionary = {}
    predicted_dictionary = {}

    for i in key_value_pair_array_gold:
        index = i.index('=')
        key = i[0:index]
        value = i[index+1:]
        gold_dictionary[key] = value

    for i in key_value_pair_array_predicted:
        index = i.index('=')
        key = i[0:index]
        value = i[index+1:]
        predicted_dictionary[key] = value

    total = 0.0
    correct = 0.0
    for key in gold_dictionary:
        total += 1.0
        if key in predicted_dictionary:
            if gold_dictionary[key] == predicted_dictionary[key]:
                correct += 1.0

    return float(correct)/total


