stat_text
=========

stat_text is a small module for performing some common tasks in statistical text processing.

## Quick Start

Let's work with a sample from Borges' [The Lottery in Babylon](http://www.class.uh.edu/mcl/faculty/armstrong/cityofdreams/texts/babylon.html):

```python
In [3]: raw_text = '''Like all the men of Babylon, I have been proconsul; like all, I have been a slave. I have known omnipotence, ignominy, imprisonment. Look here-- my right hand has no index finger. Look here--through this gash in my cape you can see on my stomach a crimson tattoo--it is the second letter, Beth. On nights when the moon is full, this symbol gives me power over men with the mark of Gimel, but it subjects me to those with the Aleph, who on nights when there is no moon owe obedience to those marked with the Gimel. In the half-light of dawn, in a cellar, standing before a black altar, I have slit the throats of sacred bulls. Once, for an entire lunar year, I was declared invisible--I would cry out and no one would heed my call, I would steal bread and not be beheaded. I have known that thing the Greeks knew not--uncertainty. In a chamber of brass, as I faced the strangler's silent scarf, hope did not abandon me; in the river of delights, panic has not failed me. Heraclides Ponticus reports, admiringly, that Pythagoras recalled having been Pyrrhus, and before that, Euphorbus, and before that, some other mortal; in order to recall similar vicissitudes, I have no need of death, nor even of imposture.'''
```

First, let's clean up this text a bit by removing punctuation, and converting everything to lowercase:

```python
In [4]: clean_text = StatText.simplify(raw_text)

In [5]: clean_text
Out[5]: u'like all the men of babylon i have been proconsul like all i have been a slave i have known omnipotence ignominy imprisonment look here my right hand has no index finger look herethrough this gash in my cape you can see on my stomach a crimson tattooit is the second letter beth on nights when the moon is full this symbol gives me power over men with the mark of gimel but it subjects me to those with the aleph who on nights when there is no moon owe obedience to those marked with the gimel in the halflight of dawn in a cellar standing before a black altar i have slit the throats of sacred bulls once for an entire lunar year i was declared invisiblei would cry out and no one would heed my call i would steal bread and not be beheaded i have known that thing the greeks knew notuncertainty in a chamber of brass as i faced the stranglers silent scarf hope did not abandon me in the river of delights panic has not failed me heraclides ponticus reports admiringly that pythagoras recalled having been pyrrhus and before that euphorbus and before that some other mortal in order to recall similar vicissitudes i have no need of death nor even of imposture'
```

Now let's compute the 1-, 2-, and 3-gram entropies:

```python
In [6]: text = StatText(clean_text)

In [7]: text.entropy(1)
Out[7]: 4.0790312417996599

In [8]: text.entropy(2)
Out[8]: 7.1996723319757852

In [9]: text.entropy(3)
Out[9]: 8.9172785811961219
```

We can also compute conditional entropy:

```python
In [13]: text.entropy(prefix='b', length=1)
Out[13]: 2.4014473774842413

In [14]: text.entropy(prefix='ba', length=1)
Out[14]: 1.0

In [15]: text.entropy(prefix='bab', length=1)
Out[15]: -0
```

The reason that the last entropy is 0 is that 'bab' is always followed by 'y' in the text. We can see this directly by computing the conditional distributions:

```python
In [17]: text.distribution(prefix='b', length=1)
Out[17]: 
{u'a': 0.086956521739130432,
 u'e': 0.47826086956521741,
 u'j': 0.043478260869565216,
 u'l': 0.086956521739130432,
 u'o': 0.043478260869565216,
 u'r': 0.086956521739130432,
 u'u': 0.13043478260869565,
 u'y': 0.043478260869565216}

In [18]: text.distribution(prefix='ba', length=1)
Out[18]: {u'b': 0.5, u'n': 0.5}

In [19]: text.distribution(prefix='bab', length=1)
Out[19]: {u'y': 1.0}
```

Now let's play around with a Markov model for the text. Let's generate some text samples from the 2-, 3-, 4-, and 5-gram models:

```python
In [21]: text.markov(2).generate_text(100)
Out[21]: u's te ibugirtu o hot ser abee hofof ss leronider d hato ire ouss re the mbe inin okno thondave busi r'

In [22]: text.markov(3).generate_text(100)
Out[22]: u'ed nowe red ordes slike des simsomnicus se witunar thostras ree me hanglepor youll bre omnic he ris '

In [23]: text.markov(4).generate_text(100)
Out[23]: u'r i having beheaded me powe on i was i have no in tattooit scarf hope did not aband been with no nee'

In [24]: text.markov(5).generate_text(100)
Out[24]: u'it the all similar vicissitudes ponticus recall i would steal bread and not be been proconsul like a'
```

We can also compute the probability that a given Markov model would emit a particular string:

```python
In [31]: text.markov(4).string_probability('four score and seven ye')
Out[31]: 9.5238095238095418e-86

In [32]: text.markov(4).string_probability('it was the best of time')
Out[32]: 4.6153846153846739e-73
```

These probabilities are usually very, very small for natural text, and for long samples there's a risk of underflow. So you have the option of computing the log probabilities instead:

```python
In [33]: text.markov(4).string_probability('four score and seven ye', log=True)
Out[33]: -195.76852306866331

In [34]: text.markov(4).string_probability('it was the best of time', log=True)
Out[34]: -166.55931658380476
```

That's pretty much it for now! If you'd like to see a slightly meatier application of the module, take a look at this blog post about [identifying authors with Markov chains](http://newdatascientist.com/blog/2013/08/28/author-identification-with-markov-models/).
