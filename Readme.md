# Natural Language Processing techniques for the analysis of xml-formatted text collections on Bulgarian.

*Licensed under GNU General Public License v3.0 (GPL-3.0).*

*Written by D. Kienzler.*

## How to use

### Preparation

Clone all files in the repository to a folder on your computer. Create a conda environment with all necessary dependencies with `conda env create -f linux_env.yml` (Linux) or `conda env create -f win_env.yml` (Windows) and activate it.

### Using the script

Navigate to the folder contanining the files, start an interactive python session and run the script with:

`python -i text_analysis.py`

Create and assign to a variable an instance of the class with:

`exmpl = TextAnalysis()`

Then, load the sample text file with: 

`corpus = exmpl.load_database()` 

Preprocess the text with:

`clean_corpus = exmpl.preprocess(corpus)`

Next, perform vectorizing with either the BoW or TF-IDF approach (or both, but assign to different variables). It is important to note that here two values are returned: 

`matrix, terms = exmpl.bow_vectorize(clean_corpus)` 

or

`matrix, terms = exmpl.tfidf_vectorize(clean_corpus)` 

Optionally plot the top ten terms of a review on a horizontal bar graph with (the last parameter is the number of the review):

`exmpl.plot_review_terms(matrix, terms, 42)`

Optionally print the C_V coherence scores of a model with a range of numbers of topics. This function calls three other functions in a loop from 1 to the maximum number of topics chosen (The first parameter is an identifier for which semantic analysis technique to use. It must be either 'lsa' or 'lda'. An optional parameter is `max_topic`, an integer denoting the highest number of topics to compare, default is 30):

`exmpl.compare_coherence('lda', matrix, terms, clean_corpus, max_topic=40)`

Create one or more semantic analysis model and array with the preferred number of topics. Again, two values are returned here. (The default value for `no_topics` is 15 for LSA ans 21 for LDA):

`model, array = exmpl.lsa_analysis(matrix, no_topics=16)`

or

`model, array = exmpl.lda_analysis(matrix, no_topics=22)`

Create a list of top words for the model's topics:

`top_terms = exmpl.top_terms(model, terms)`

Represent the topics' occurence probability as horizontal bar graph for the model:

`exmpl.plot_model(array, top_terms)`

### For full documentation use:

`help(TextAnalysis)`

or for a specific command

`help(TextAnalysis.load_database)`


