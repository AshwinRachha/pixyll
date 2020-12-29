

```
!pip install bert-extractive-summarizer
```

    Requirement already satisfied: bert-extractive-summarizer in /usr/local/lib/python3.6/dist-packages (0.6.1)
    Requirement already satisfied: scikit-learn in /usr/local/lib/python3.6/dist-packages (from bert-extractive-summarizer) (0.22.2.post1)
    Requirement already satisfied: spacy in /usr/local/lib/python3.6/dist-packages (from bert-extractive-summarizer) (2.2.4)
    Requirement already satisfied: transformers in /usr/local/lib/python3.6/dist-packages (from bert-extractive-summarizer) (4.1.1)
    Requirement already satisfied: scipy>=0.17.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.4.1)
    Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.0.0)
    Requirement already satisfied: numpy>=1.11.0 in /usr/local/lib/python3.6/dist-packages (from scikit-learn->bert-extractive-summarizer) (1.19.4)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (0.4.1)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (0.8.0)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (4.41.1)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (51.0.0)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (2.0.5)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (2.23.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (1.1.3)
    Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (7.4.0)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (3.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (1.0.5)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (1.0.5)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy->bert-extractive-summarizer) (1.0.0)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (0.8)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (2019.12.20)
    Requirement already satisfied: tokenizers==0.9.4 in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (0.9.4)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (20.8)
    Requirement already satisfied: sacremoses in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (0.0.43)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers->bert-extractive-summarizer) (3.0.12)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (2020.12.5)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy->bert-extractive-summarizer) (2.10)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (3.3.0)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers->bert-extractive-summarizer) (2.4.7)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers->bert-extractive-summarizer) (7.1.2)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers->bert-extractive-summarizer) (1.15.0)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (3.4.0)
    Requirement already satisfied: typing-extensions>=3.6.4; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy->bert-extractive-summarizer) (3.7.4.3)
    


```
!pip install transformers
```

    Requirement already satisfied: transformers in /usr/local/lib/python3.6/dist-packages (4.1.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from transformers) (1.19.4)
    Requirement already satisfied: dataclasses; python_version < "3.7" in /usr/local/lib/python3.6/dist-packages (from transformers) (0.8)
    Requirement already satisfied: packaging in /usr/local/lib/python3.6/dist-packages (from transformers) (20.8)
    Requirement already satisfied: filelock in /usr/local/lib/python3.6/dist-packages (from transformers) (3.0.12)
    Requirement already satisfied: tokenizers==0.9.4 in /usr/local/lib/python3.6/dist-packages (from transformers) (0.9.4)
    Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.6/dist-packages (from transformers) (2019.12.20)
    Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.6/dist-packages (from transformers) (4.41.1)
    Requirement already satisfied: requests in /usr/local/lib/python3.6/dist-packages (from transformers) (2.23.0)
    Requirement already satisfied: sacremoses in /usr/local/lib/python3.6/dist-packages (from transformers) (0.0.43)
    Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from packaging->transformers) (2.4.7)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (3.0.4)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2.10)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (1.24.3)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests->transformers) (2020.12.5)
    Requirement already satisfied: joblib in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.0.0)
    Requirement already satisfied: click in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (7.1.2)
    Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from sacremoses->transformers) (1.15.0)
    


```
!pip install spacy
```

    Requirement already satisfied: spacy in /usr/local/lib/python3.6/dist-packages (2.2.4)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (3.0.5)
    Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.0)
    Requirement already satisfied: plac<1.2.0,>=0.9.6 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.1.3)
    Requirement already satisfied: thinc==7.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (7.4.0)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.5)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (2.0.5)
    Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.0.5)
    Requirement already satisfied: blis<0.5.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (0.4.1)
    Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (4.41.1)
    Requirement already satisfied: requests<3.0.0,>=2.13.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (2.23.0)
    Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (0.8.0)
    Requirement already satisfied: setuptools in /usr/local/lib/python3.6/dist-packages (from spacy) (51.0.0)
    Requirement already satisfied: numpy>=1.15.0 in /usr/local/lib/python3.6/dist-packages (from spacy) (1.19.4)
    Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from catalogue<1.1.0,>=0.0.7->spacy) (3.3.0)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (1.24.3)
    Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2.10)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (2020.12.5)
    Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests<3.0.0,>=2.13.0->spacy) (3.0.4)
    Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy) (3.4.0)
    Requirement already satisfied: typing-extensions>=3.6.4; python_version < "3.8" in /usr/local/lib/python3.6/dist-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy) (3.7.4.3)
    


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

    From a very early age, perhaps the age of five or six, I knew that when I grew up I should be a writer. I was the middle child of three, but there was a gap of five years on either side, and I barely saw my father before I was eight. These magazines were the most pitiful burlesque stuff that you could imagine, and I took far less trouble with them than I now 
     would with the cheapest journalism. As a very small child I used to imagine that I was, say, Robin Hood, and picture myself as the hero of thrilling adventures, but quite
       soon my ‘story’ ceased to be narcissistic in a crude way and became more and more a mere description of what I was doing and the things I saw.
    


```
GPT2_model = TransformerSummarizer(transformer_type = 'GPT2', transformer_model_key= 'gpt2-medium')
GPT2_summary = ''.join(GPT2_model(body, min_length = 100))
```

    Some weights of GPT2Model were not initialized from the model checkpoint at gpt2-medium and are newly initialized: ['h.0.attn.masked_bias', 'h.1.attn.masked_bias', 'h.2.attn.masked_bias', 'h.3.attn.masked_bias', 'h.4.attn.masked_bias', 'h.5.attn.masked_bias', 'h.6.attn.masked_bias', 'h.7.attn.masked_bias', 'h.8.attn.masked_bias', 'h.9.attn.masked_bias', 'h.10.attn.masked_bias', 'h.11.attn.masked_bias', 'h.12.attn.masked_bias', 'h.13.attn.masked_bias', 'h.14.attn.masked_bias', 'h.15.attn.masked_bias', 'h.16.attn.masked_bias', 'h.17.attn.masked_bias', 'h.18.attn.masked_bias', 'h.19.attn.masked_bias', 'h.20.attn.masked_bias', 'h.21.attn.masked_bias', 'h.22.attn.masked_bias', 'h.23.attn.masked_bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


```
print(GPT2_summary)
```

    From a very early age, perhaps the age of five or six, I knew that when I grew up I should be a writer. I was the middle child of three, but there was a gap of five years on either side, and I barely saw my father before I was eight. I had the lonely child's habit of making up stories and holding conversations with imaginary persons, and I think from the very start my literary
    ambitions were mixed up with the feeling of being isolated and undervalued. At eleven, when the war or 1914-18 broke out, I wrote a patriotic poem which 
    was printed in the local newspaper, as was another, two years later, on the death of Kitchener. The ‘story’ must, 
    I suppose, have reflected the styles of the various writers I admired at different ages, but so far as I remember it always had the same meticulous descriptive quality.
    


```

```
