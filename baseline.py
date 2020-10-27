"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # ------------------- TRAINING -------------------
    tag_freq = {}
    word_tags = {}
    # Iterate through training data to get most common tag overall and most common for each word
    for sentence in train:
        for words in sentence:
            tag = words[1]
            word = words[0]
            # Count occurences of each tag
            tag_freq[tag] = tag_freq.get(tag,0) + 1
            if word not in word_tags:
                    # Ex: word_tags: {"rabbit":{"NOUN":15,"VERB":1}}
                    tag_dict = {}
                    tag_dict[tag] = 1
                    word_tags[word] = tag_dict
            else:
                # If tag exists in word entry, increment by one, else initialize to 1 
                word_tags[word][tag] = word_tags[word].get(tag,0) + 1
    #
    # ------------------- TESTING -------------------
    # Match each word to most frequent tag for that word
    tagged_words = {}
    for word in word_tags:
        tagged_words[word] = max(word_tags[word],key=word_tags[word].get)
    # Get most common tag in training data
    common_tag = max(tag_freq,key=tag_freq.get)
    test_tagged = []
    for sentence in test:
        test_sentence = []
        for word in sentence:
            if word in test_sentence:
                continue
            elif word in tagged_words:
                word_tag = tagged_words[word]
                test_sentence.append((word,word_tag))
            else:
                test_sentence.append((word,common_tag))
        test_tagged.append(test_sentence)
    # Return tagged test set
    return test_tagged