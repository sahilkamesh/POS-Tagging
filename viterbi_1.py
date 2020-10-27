"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    '''
    Given an input word sequence W, our goal is to find the tag sequence T that maximizes
    
    log[P(T|W)] ∝ ∑(i=1 to n){log[P(w_i|t_i)]} + ∑(k=1 to n){log[P(t_k|t_(k-1))]}

    P_t(t_k|t_k−1) transition probabilities, P_e(w_i|t_i) emission probabilities
    '''
    # ------------------- ESTIMATION -------------------
    # Use helper function "counting" to count occurrences of tags, tag pairs, tag/word pairs.
    end = math.ceil(len(train)/4)
    #[tag_count,transition_count,emission_count] = counting(train[0:end])
    [tag_count,transition_count,emission_count] = counting(train)

    # Use helper function "estimation" to estimate transition and emission probabilities
    [p_t,p_e] = estimation(tag_count,transition_count,emission_count) 

    # ------------------- DECODING - VITERBI ALGORITHM -------------------
    #m = len(tag_count)
    print("\n-------- DECODING --------\n")    
    output = []
    tags = []
    for tag in tag_count.keys():
        if tag not in ["START","END"]:
            tags.append(tag)
    print("Tags: ",tags,"\n")
    for sentence in test:
        n = len(sentence)
        # Initialize Trellis
        trellis = []
        # Fill first column with initial probabilities
        temp = []
        word1 = sentence[0]
        for tag in tags:
            if tag not in p_t["START"]:
                ps = p_t["START"]["UNKNOWN"]
            else:                
                ps = p_t["START"][tag]
            if word1 not in p_e:
                pe = p_e["UNKNOWN"][tag]
            elif tag not in p_e[word1]:
                pe = p_e[word1]["UNKNOWN"]
            else:
                pe = p_e[word1][tag]
            temp.append((ps+pe,tag))
        trellis.append(temp)
        for k in range(1,n):
            temp = []
            word = sentence[k]
            for tag1 in tags:
                t1 = tags.index(tag1)
                for tag2 in tags:
                    t2 = tags.index(tag2)
                    prev_v = trellis[k-1][t1][0]
                    if tag2 not in p_t[tag1]:
                        pt = p_t[tag1]["UNKNOWN"]
                    else:
                        pt = p_t[tag1][tag2]
                    if word not in p_e:
                        pe = p_e["UNKNOWN"][tag1]
                    elif tag1 not in p_e[word]:
                        pe = p_e[word]["UNKNOWN"]
                    else:
                        pe = p_e[word][tag1]
                    # current v
                    curr_v = prev_v + pt + pe
                    curr_tuple = (curr_v,tag1)
                    if t1 == 0:
                        temp.append(curr_tuple)
                    elif temp[t2][0] < curr_v:
                        temp[t2] = curr_tuple
            trellis.append(temp)       
        # Finished the trellis, pick the best tag in the final column
        output_tags = [] 
        end = trellis[len(trellis)-1]
        max_tag_idx = end.index((max(end, key=lambda pair: pair[0])))
        output_tags.append(tags[max_tag_idx])

        prev_tag = max(end, key=lambda pair: pair[0])
        for i in range(len(trellis)-1, 0, -1):
            prev_tag = trellis[i - 1][tags.index(prev_tag[1])]
            output_tags.insert(0, prev_tag[1])

        max_start_tag = max(trellis[0], key=lambda pair: pair[0])[1]
        output_tags[0] = max_start_tag

        output_sentence = []
        for word_idx in range(len(sentence)):
            output_sentence.append((sentence[word_idx],output_tags[word_idx]))

        output.append(output_sentence)

    return output

'''
Counts occurrences of tags, tag pairs, tag/word pairs, given some train data
'''
def counting(train):
    print("\n-------- COUNTING --------\n")
    tag_count = {}
    transition_count = {}
    emission_count = {}
    for sentence in train:
        #tag_count["START"] = tag_count.get("START",0) + 1
        for word_idx in range(len(sentence)-1):
            word = sentence[word_idx][0]
            tag = sentence[word_idx][1]
            # (i) count tag occurences
            tag_count[tag] = tag_count.get(tag,0) + 1
            # (ii) count tag transitions - # ex: {"NOUN_VERB":15,"NOUN_ADV":14}
            next_tag = sentence[word_idx+1][1]
            if tag not in transition_count:
                transition_count[tag] = {}
            curr_count = transition_count[tag]
            curr_count[next_tag] = curr_count.get(next_tag,0) + 1
            # (iii) count word tag pairs
            if word not in emission_count:
                emission_count[word] = {}
            word_tags = emission_count[word]
            word_tags[tag] = word_tags.get(tag,0) + 1
    return [tag_count,transition_count,emission_count]

'''
Estimates transition and emission probabilities, given counts
Smoothes the probabilities and takes the log
'''
def estimation(tag_count,transition_count,emission_count):
    print("\n-------- ESTIMATION --------\n")
    # (i) estimation transition probabilities P(t_k|t_(k-1))
    '''
        P(t_k|t_(k-1)) = [count(t_(k-1),t_k)+α]/[n+α(V+1)]. Where, 
        α = Laplace Parameter = smoothing 
        n = count(t_(k-1)) = tag_count[prev_tag]
        V = number of tags = len(tag_count)
    '''
    smoothing = 0.0000001       # Set Laplace parameter
    p_t = {}                # Dictionary to store tag pair probabilities
    v = len(tag_count)
    max_n = 0
    for tag in transition_count:
        p_t[tag] = {}
        n = tag_count[tag]
        for tag2 in transition_count[tag]:
            count = transition_count[tag][tag2]
            p_t[tag][tag2] = math.log((count + smoothing)/(n + smoothing*v)) 
            # "UNKNOWN" denotes all the words we have not encountered in training data
        p_t[tag]["UNKNOWN"] = math.log(smoothing/(n + smoothing*v))

    #
    # (ii) estimation emission probabilities P_e(w_i|t_i)
    '''
        P_e(w_i|t_i) = [count(t_i,w_i) + α]/[n + α(V+1)]. Where, 
        α = Laplace Parameter = smoothing 
        n = count(t_i) = tag_count[tag]
        V = number of tags = len(tag_count)
    '''
    # Emission count = {'DET':{'the': 4362, 'an': 123}}
    smoothing = 0.0000001     # Set Laplace parameter
    p_e = {}                # Dictionary to store tag pair probabilities
    for word in emission_count:
        p_e[word] = {}
        total = 0
        for tag in emission_count[word]:
            n = tag_count[tag]
            count = emission_count[word][tag]
            total += count
            p_e[word][tag] = math.log((count + smoothing)/(n + smoothing*(v+1))) 
        p_e[word]["UNKNOWN"] = math.log((smoothing)/(total + smoothing*(v+1)))
    p_e["UNKNOWN"] = {}
    max_n = 0
    for tag in tag_count:
        n = tag_count[tag]
        if n>max_n:
            max_n = n
        # "UNKNOWN" denotes all the words we have not encountered in training data
        p_e["UNKNOWN"][tag] = math.log(smoothing/(n + smoothing*(v+1)))
    p_e["UNKNOWN"]["UNKNOWN"] = math.log(smoothing/(max_n + smoothing*(v+1)))
    # Return transition & emission probabilities
    return [p_t,p_e]
