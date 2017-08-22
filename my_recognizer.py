import warnings
from asl_data import SinglesData
import sys

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # TODO implement the recognizer
    # return probabilities, guesses
    all_words = models.keys()
    for i in range(0, test_set.num_items):
      logL_dict = {}
      X, lengths = test_set.get_item_Xlengths(i)
      max_logL = -sys.maxsize - 1
      max_word = None
      for word in all_words:
        try:
          logL = models[word].score(X, lengths)
          logL_dict[word] = logL
          if logL > max_logL:
              max_logL = logL
              max_word = word
        except:
          logL_dict[word] = - sys.maxsize - 1
      probabilities.append(logL_dict)
      guesses.append(max_word)
    return probabilities, guesses
    
