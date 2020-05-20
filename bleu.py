from nltk.translate.bleu_score import sentence_bleu

ref_path = 'target.txt'
hyp_path = 'generated.txt'


def load_data(path):
    result = []
    with open(path, 'r') as f:
        lines = f.readlines()  
    
        for line in lines:
            line_data = list(map(str, line.strip().split(' ')))
            for read_data in line_data:
                result.append(read_data)
    return result

def cal_bleu(ref_path, hyp_path):
    print('Calculating BLEU...')
    references = load_data(ref_path)
    hypothesis = load_data(hyp_path)
    print('reference_num:', len(references))
    print('candidate_num:', len(hypothesis))
    reference, candidate = [references], hypothesis

    bleu1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4

    



if __name__ == "__main__":
    bleu1, bleu2, bleu3, bleu4 = cal_bleu(ref_path, hyp_path)
    print('BLEU_1(Cumulative 1-gram): %f' % bleu1)
    print('BLEU_2(Cumulative 2-gram): %f' % bleu2)
    print('BLEU_3(Cumulative 3-gram): %f' % bleu3)
    print('BLEU_4(Cumulative 4-gram): %f' % bleu4)