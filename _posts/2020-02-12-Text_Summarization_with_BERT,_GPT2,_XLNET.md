---
layout:     post
title:      Please cut it short for me!
date:       2020-07-12 12:32:18
summary:    Extractive summarization with BERT and GPT2.
categories: jekyll pixyll
---

```
!pip install bert-extractive-summarizer
```
    
```
!pip install transformers
```

```
!pip install spacy
```

 

```
from summarizer import Summarizer, TransformerSummarizer
```


```
body = '''

From a very early age, perhaps the age of five or six, I knew that when I grew up I should be a writer. Between the
ages of about seventeen and twenty-four I tried to abandon this idea, but I did so with the consciousness that I was
outraging my true nature and that sooner or later I should have to settle down and write books.
I was the middle child of three, but there was a gap of five years on either side, and I barely saw my father before I was eight. 
For this and other reasons I was somewhat lonely, and I soon developed disagreeable mannerisms which made me unpopular throughout my schooldays. 
I had the lonely child's habit of making up stories and holding conversations with imaginary persons, and I think from the very start my literary
ambitions were mixed up with the feeling of being isolated and undervalued. I knew that I had a facility with words and a power of facing unpleasant facts,
and I felt that this created a sort of private world in which I could get my own back for my failure in everyday life. Nevertheless the volume of serious — 
i.e. seriously intended — writing which I produced all through my childhood and boyhood would not amount to half a dozen pages. I wrote my first poem at the
age of four or five, my mother taking it down to dictation. I cannot remember anything about it except that it was about a tiger and the tiger had ‘chair-like teeth’ 
— a good enough phrase, but I fancy the poem was a plagiarism of Blake's ‘Tiger, Tiger’. At eleven, when the war or 1914-18 broke out, I wrote a patriotic poem which 
was printed in the local newspaper, as was another, two years later, on the death of Kitchener. From time to time, when I was a bit older, I wrote bad and
usually unfinished ‘nature poems’ in the Georgian style. I also attempted a short story which was a ghastly failure. That was the total of the would-be 
serious work that I actually set down on paper during all those years.
However, throughout this time I did in a sense engage in literary activities. To begin with there was the made-to-order stuff which I produced quickly, 
easily and without much pleasure to myself. Apart from school work, I wrote vers d'occasion, semi-comic poems which I could turn out at what now seems to 
me astonishing speed — at fourteen I wrote a whole rhyming play, in imitation of Aristophanes, in about a week — and helped to edit a school magazines, both
printed and in manuscript. These magazines were the most pitiful burlesque stuff that you could imagine, and I took far less trouble with them than I now 
would with the cheapest journalism. But side by side with all this, for fifteen years or more, I was carrying out a literary exercise of a quite different 
kind: this was the making up of a continuous ‘story’ about myself, a sort of diary existing only in the mind. I believe this is a common habit of children
and adolescents. As a very small child I used to imagine that I was, say, Robin Hood, and picture myself as the hero of thrilling adventures, but quite
soon my ‘story’ ceased to be narcissistic in a crude way and became more and more a mere description of what I was doing and the things I saw. For minutes 
at a time this kind of thing would be running through my head: ‘He pushed the door open and entered the room. A yellow beam of sunlight, filtering through 
the muslin curtains, slanted on to the table, where a match-box, half-open, lay beside the inkpot. With his right hand in his pocket he moved across to the 
window. Down in the street a tortoiseshell cat was chasing a dead leaf’, etc. etc. This habit continued until I was about twenty-five, right through my 
non-literary years. Although I had to search, and did search, for the right words, I seemed to be making this descriptive effort almost against my will,
under a kind of compulsion from outside. The ‘story’ must, 
I suppose, have reflected the styles of the various writers I admired at different ages, but so far as I remember it always had the same meticulous descriptive quality.

        '''
```


```
bert_model = Summarizer()

bert_summary = ''.join(bert_model(body, min_length = 100))

print(bert_summary)
```

    From a very early age, perhaps the age of five or six, I knew that when I grew up I should be a writer.
    I was the middle child of three, but there was a gap of five years on either side, and I barely saw my 
    father before I was eight. These magazines were the most pitiful burlesque stuff that you could imagine,
    and I took far less trouble with them than I now would with the cheapest journalism. As a very small child
    I used to imagine that I was, say, Robin Hood, and picture myself as the hero of thrilling adventures, but 
    quite soon my ‘story’ ceased to be narcissistic in a crude way and became more and more a mere description 
    of what I was doing and the things I saw.
    


```
GPT2_model = TransformerSummarizer(transformer_type = 'GPT2', transformer_model_key= 'gpt2-medium')

GPT2_summary = ''.join(GPT2_model(body, min_length = 100))
print(GPT2_summary)
```

    From a very early age, perhaps the age of five or six, I knew that when I grew up I should be a writer.
    I was the middle child of three, but there was a gap of five years on either side, and I barely saw my father
    before I was eight. I had the lonely child's habit of making up stories and holding conversations with imaginary
    persons, and I think from the very start my literary ambitions were mixed up with the feeling of being isolated 
    and undervalued. At eleven, when the war or 1914-18 broke out, I wrote a patriotic poem which was printed in the
    local newspaper, as was another, two years later, on the death of Kitchener. The ‘story’ must,  I suppose, have 
    reflected the styles of the various writers I admired at different ages, but so far as I remember it always had 
    the same meticulous descriptive quality.
    


```

```
