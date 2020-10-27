# POS-Tagging
This is an implementation of part of speech (POS) tagging using an HMM model.  
This was done in the CS 440 (Artificial Intelligence) class at the University of Illinois at Urbana-Champaign (UIUC).

This model is built using training data from the [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus). 

We generate a baseline model that only uses word-tag frequencies to predict tags - this achieves a ~94% accuracy on the Brown development data.

Next, we build a Hidden Markov Model that uses context (previous tag) as well as word-tag frequencies to make a prediction. 
This implementation is done using the **Viterbi** algorithm. This achieves a ~93% accuracy, but improves the multitag accuracy from the baseline model (~93%).

To run it:  

```python3 mp4.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm [baseline, viterbi_1]```  
  -- train     : Training data file (.txt)  
  -- test      : Testing data file  (.txt)  
  -- algorithm : Which algorithm to use  
  
