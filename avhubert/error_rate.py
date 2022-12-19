import editdistance
import numpy as np
import torch

"""Compute error rates for word and character.

    Adopted from "Leveraging Self-supervised Learning for AVSR"
    author: Xichen Pan
"""

def compute_error_word(predictionStr, targetStr):
    predictionList = predictionStr.split('\n')
    targetList = targetStr.split('\n')
    totalEdits = 0
    totalChars = 0
    for pred, trgt in zip(predictionList, targetList):
        pred_words, trgt_words = pred.strip().split(), trgt.strip().split()
        numEdits = editdistance.eval(pred_words, trgt_words)
        totalEdits = totalEdits + numEdits
        totalChars = totalChars + len(trgt_words)

    return totalEdits, totalChars

# if __name__=='__main__':
#     predictionStr = "the day that there's no more\n but \n so there i came in up with s\n 2500 million \n if we did we could see\n they found themselves"
#     targetStr = "the day that there's no more waste bread to be brewed is the day that we can shut up shop and\n no i didn't say iphone i said a flip camera google that they are circa 2006 and i'd go\n so there i am in the embassy in beijing off to the great hall of the people with our ambassador who had asked me\n and more than 500 million extra minutes of time after school has been spent learning\n if we did we could see that our own resources are easier to use than anybody\n they found themselves forced to abandon their nomadic lives forced to make contact with their aw neighbors"
#     we, wc = compute_error_word(predictionStr, targetStr)
#     print(we, wc)


def compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx):
    """
    Function to compute the Word Error Rate using the Predicted character indices and the Target character
    indices over a batch. The words are obtained by splitting the output at spaces.
    WER is computed by dividing the total number of word edits (computed using the editdistance package)
    with the total number of words (total => over all the samples in a batch).
    The <EOS> token at the end is excluded before computing the WER. Words with only a space are removed as well.
    """

    totalEdits, totalWords = compute_error_word(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, spaceIx)

    return totalEdits / totalWords
