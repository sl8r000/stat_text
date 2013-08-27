import collections
import numpy as np
import pandas as pd
import unicodedata
import sys


# To avoid writing "lambda x: value" in several places.
def return_value(value):
    return lambda: value


class TextMarkovModel(object):
    def __init__(self, initial_disribution, conditional_distribution, fake_zero=None):
        self.n_gram_length = len(initial_disribution.keys()[0])
        self.fake_zero = fake_zero if fake_zero is not None else 0.000001
        self.initial_disribution = collections.defaultdict(return_value(self.fake_zero), initial_disribution)
        self.conditional_distribution = conditional_distribution
        for prefix in self.conditional_distribution:
            self.conditional_distribution[prefix] = collections.defaultdict(return_value(self.fake_zero),
                                                                            self.conditional_distribution[prefix])
        self.conditional_distribution = collections.defaultdict(
            return_value(collections.defaultdict(return_value(self.fake_zero))), conditional_distribution)

    def sample_from_distribution(self, distribution):
        keys = distribution.keys()
        values = np.array(distribution.values())

        if len(keys) == 0:
            return self.fake_zero

        sample_index = values.cumsum().searchsorted(np.random.uniform())
        if sample_index >= len(keys):
            return keys[-1]
        return keys[sample_index]

    def string_probability(self, input_string, log=False):
        if len(input_string) < self.n_gram_length:
            raise Exception('Input string length is {}, which is less than this model\'s length of {}'
                            .format(len(input_string), self.n_gram_length))

        char_probabilities = np.zeros(len(input_string) - self.n_gram_length + 1)
        char_probabilities[0] = self.initial_disribution[input_string[0:self.n_gram_length]]
        for i in xrange(1, len(input_string) - self.n_gram_length + 1):
            text_segment = input_string[i:i+self.n_gram_length]
            prefix = text_segment[0:-1]
            last_character = text_segment[-1]
            char_probabilities[i] = self.conditional_distribution[prefix][last_character]

        if log:
            return np.sum(np.log(char_probabilities))
        else:
            return np.exp(np.sum(np.log(char_probabilities)))

    def generate_text(self, size):
        sample = self.sample_from_distribution(self.initial_disribution)
        chars = [c for c in sample]
        prefix = sample[1:]
        for i in xrange(size - self.n_gram_length):
            distribution = self.conditional_distribution[prefix]
            next_char = self.sample_from_distribution(distribution)
            chars.append(next_char)
            prefix = prefix[1:] + next_char

        return ''.join(chars)


class StatText(object):
    def __init__(self, text):
        self.text = text

    @classmethod
    def simplify(cls, text):
        text = unicode(text)
        kill_symbols_punctuation_digits = dict((char_number, None) for char_number in xrange(sys.maxunicode)
                                               if unicodedata.category(unichr(char_number))[0] in ['P', 'S', 'N'])

        nice_character_text = text.translate(kill_symbols_punctuation_digits).lower()
        return ' '.join(nice_character_text.split())

    def get_ngrams_list(self, length=1, prefix=None):
        text_length = len(self.text)
        if prefix is None:
            n_grams = [self.text[i:i+length] for i in range(text_length - length)]
        else:
            prefix_length = len(prefix)
            n_gram_length = length + prefix_length
            n_grams = [self.text[i:i+n_gram_length] for i in range(text_length - n_gram_length)
                       if self.text[i:i+prefix_length] == prefix]

        return n_grams

    def entropy(self, length=1, prefix=None):
        text_length = len(self.text)
        n_grams = self.get_ngrams_list(length, prefix)

        frequency_counts = np.array(collections.Counter(n_grams).values())

        if prefix is None:
            normalizer = float(text_length - length)
        else:
            normalizer = float(sum(frequency_counts))

        probabilities = frequency_counts / normalizer
        log_probabilities = np.log2(probabilities)

        return -np.dot(probabilities, log_probabilities)

    def distribution(self, length=1, prefix=None):
        n_grams = self.get_ngrams_list(length, prefix)
        if prefix is not None:
            prefix_length = len(prefix)
            n_grams = [n_gram[prefix_length:] for n_gram in n_grams]

        frequency_counts = pd.Series(collections.Counter(n_grams))
        normalizer = float(sum(frequency_counts))
        probability_series = frequency_counts / normalizer

        return probability_series.to_dict()

    def complete_conditional_distribution(self, length=1, prefix_length=1):
        n_grams = self.get_ngrams_list(length=length + prefix_length)

        probabilities = collections.defaultdict(collections.Counter)
        for n_gram in n_grams:
            prefix = n_gram[:prefix_length]
            suffix = n_gram[prefix_length:]

            probabilities[prefix][suffix] += 1

        for prefix in probabilities:
            normalizer = float(np.sum(probabilities[prefix].values()))
            probabilities[prefix] = pd.Series(probabilities[prefix]) / normalizer
            probabilities[prefix] = probabilities[prefix].to_dict()

        return probabilities

    def markov(self, length=2):
        initial_distribution = self.distribution(length=length)
        conditional_distribution = self.complete_conditional_distribution(length=1, prefix_length=length-1)

        return TextMarkovModel(initial_distribution, conditional_distribution)
