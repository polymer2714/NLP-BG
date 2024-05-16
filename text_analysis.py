"""
Natural Language Processing techniques for the analysis of xml-formatted text collections on Bulgarian.

Licensed under GNU General Public License v3.0 (GPL-3.0)

Written by D. Kienzler.
"""

# Import library for database import.

from xml.dom.minidom import parse


# Import libraries for preprocessing.

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
import simplemma


# Import libraries for text vectorizing.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Import libraries for topic modeling.

from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


# Import library for graphical representation.

import matplotlib.pyplot as plt


# Downlad nltk corpora.

import nltk
nltk.download('punkt')


# Define class and methods.

class TextAnalysis():
    """
    A class for using Natural Language Processing techniques for the analysis of xml-formatted text collections on Bulgarian.

    Methods:

        load_database(self,file='Customer_feedback_bg_dataset.xml', tag_name='Review'):
            Import a text corpus and return the reviews as string elements of a list.
    
        preprocess(self, corpus):
            Perform several preprocessing steps to achieve a cleaned corpus.
    
        bow_vectorize(self, clean_corpus):
            Create a bag-of-words token count document-term matrix.
    
        tfidf_vectorize(self, clean_corpus):
            Create a TF-IDF document-term matrix.

        plot_review_terms(self, matrix, terms, review_no):
            Plot the 10 most relevant words of a review on a horizontal bar graph.
    
        lsa_analysis(self, matrix, no_topics=19):
            Create a Latent Semantic Analysis model and a by Singular Value Decomposition truncated matrix (array) from the original document-term matrix.
    
        lda_analysis(self, matrix, no_topics=19):
            Create a model and array according to Latent Dirichlet Allocation probability distributions from the original document-term matrix.
    
        top_terms(self, model, terms, no_terms=20):
            Extract the top terms per topic by relevance.
    
        get_coherence(self, top_terms, clean_corpus):
            Calculate the C_V coherence score for a topic model.
    
        compare_coherence(self, mod_alg, matrix, terms, clean_corpus, max_topic=30):
            Print the C_V coherence scores for a topic model with differing numbers of topics.
    
        plot_model(self, array, top_terms):
            Represent the topic occurence probabilities of a model visually in a horizontal bar chart.

    Example usage:

        The following code will calculate all models (all possible combinations of BoW/TFIDF and LSA/LDA) with the number of topics that have the highest C_V value and are more than ten topics, each model's respective coherence scores from 1 to 30 topics, and each model's top terms. It will also plot the relevance of the ten top words of topic 1 for both BoW and TFIDF vectorization. Finally, it will represent each model as a horizontal bar graph plot showing the topic occurence probability.

        Instead of calculating all models, it is possible to choose only code lines relevant to one or several specific models.

        # Instantiate object of class TextAnalysis.
        txt_anls = TextAnalysis()
        
        # Load database and store in variable.
        corpus = txt_anls.load_database()
        
        # Preprocess the text data and store in variable.
        clean_corpus = txt_anls.preprocess(corpus)
        
        # Create a BoW or TFIDF document-term matrix and an array of related terms and store in separate variables.
        bow_matrix, bow_terms = txt_anls.bow_vectorize(clean_corpus)
        tfidf_matrix, tfidf_terms = txt_anls.tfidf_vectorize(clean_corpus)

        # Optional: Plot the ten most relevant words for a review as horizontal bar graph.
        txt_anls.plot_review_terms(bow_matrix, bow_terms, 1)
        txt_anls.plot_review_terms(tfidf_matrix, tfidf_terms, 1)
        
        # Optional: Compare C_V coherence scores of models with different numbers of topics.
        txt_anls.compare_coherence('lsa', bow_matrix, bow_terms, clean_corpus)
        txt_anls.compare_coherence('lsa', tfidf_matrix, tfidf_terms, clean_corpus)
        txt_anls.compare_coherence('lda', bow_matrix, bow_terms, clean_corpus)
        txt_anls.compare_coherence('lda', tfidf_matrix, tfidf_terms, clean_corpus)
        
        # Create different semantic analysis models and document-topic matrixes (arrays) with preferred number of topics and store in seperate variables.
        lsa_bow_model, lsa_bow_array = txt_anls.lsa_analysis(bow_matrix, no_topics=16)
        lsa_tfidf_model, lsa_tfidf_array = txt_anls.lsa_analysis(tfidf_matrix, no_topics=11) 
        lda_bow_model, lda_bow_array = txt_anls.lda_analysis(bow_matrix, no_topics=12) 
        lda_tfidf_model, lda_tfidf_array = txt_anls.lda_analysis(tfidf_matrix,  no_topics=19) 
        
        # Create and store in a list the top words of the models' topics.
        lsa_bow_top = txt_anls.top_terms(lsa_bow_model, bow_terms)
        lsa_tfidf_top = txt_anls.top_terms(lsa_tfidf_model, tfidf_terms)
        lda_bow_top = txt_anls.top_terms(lda_bow_model, bow_terms)
        lda_tfidf_top = txt_anls.top_terms(lda_tfidf_model, tfidf_terms) 
        
        # Represent visually the occurence probability of different topics in the models.
        txt_anls.plot_model(lsa_bow_array, lsa_bow_top)
        txt_anls.plot_model(lsa_tfidf_array, lsa_tfidf_top)
        txt_anls.plot_model(lda_bow_array, lda_bow_top)
        txt_anls.plot_model(lda_tfidf_array, lda_tfidf_top)
    """

    def load_database(self,file='Customer_feedback_bg_dataset.xml', tag_name='Review'):
        """
        Import a text corpus and return the reviews as string elements of a list.

        Parameters:
            file (str, opt.): Xml-file as input, default is the sample text. 
            tag_name (str, opt.): Tag name for extracting relevant fields from xml-file, default is 'Review'.

        Return value:
            corpus (list): List of reviews as strings.
        """
        database = parse(file)
        corpus = [review.firstChild.wholeText for review in database.getElementsByTagName(tag_name)]
        return corpus
    
    def preprocess(self, corpus):
        """
        Perform several preprocessing steps to achieve a cleaned corpus.

        The preprocessing steps include: Removal of punctuation and words containing non-cyrillic characters or digits, conversion to lowercase, tokenization, spellchecking and -correcting, and lemmatization.

        Parameter:
            corpus (list): Return value of method load_database().

        Return value:
            clean_corpus (list): List of preprocessed reviews as strings.
        """
        
        # Remove punctuation and latin characters.
        pre_corpus = [re.sub(r'[/.-]',' ',review) for review in corpus]
        pre_corpus = [re.sub('[a-zA-Z]',' ',review) for review in corpus]
        
        # Convert to lowercase lists of tokens.
        pre_corpus = [word_tokenize(review.lower()) for review in pre_corpus]

        # Create a Bulgarian word frequency list for spell-checking.
        spell = SpellChecker(language=None, distance=1)
        spell.word_frequency.load_text_file('bulgarian.txt')

        # Review missing words in dictionary and add alphabetical words to dictionary.
        unknown = [spell.unknown(review) for review in pre_corpus if spell.unknown(review) != set()]
        missing = [[w for w in review  if spell.correction(w) == None] for review in unknown]
        missing_list = []
        for i in missing:
            for j in i:
                if j.isalpha():
                    missing_list.append(j)
        spell.word_frequency.load_words(missing_list)

        # Spell correction and removal of non-alphabetical terms.
        pre_corpus = [[spell.correction(t) for t in review if t.isalpha()] for review in pre_corpus]

        # Removal of stopwords and misspellings without corrections.
        stop_words = open('stopwords-bg.txt', encoding='utf8').read()
        pre_corpus = [[t for t in review if t != None and t not in stop_words] for review in pre_corpus]

        # Lemmatizing, i.e., removing the inflexional endings of word variations to be left with dictionary forms of words.
        pre_corpus = [[simplemma.lemmatize(t, lang='bg', greedy=True) for t in review] for review in pre_corpus] 
        
        # Return a list of strings for input into vectorizers.
        clean_corpus = [''.join(t + ' ' for t in review) for review in pre_corpus]
        return clean_corpus

    def bow_vectorize(self, clean_corpus):
        """
        Create a bag-of-words token count document-term matrix.

        Parameter:
            clean_corpus (list): Return value of method preprocess().

        Return values:
            bow_matrix (scipy.sparse._csr.csr_matrix): Token count document-term matrix.
            bow_terms (numpy.ndarray): Array of term names.
        """
        
        # Create a token count document-term matrix from the training data.
        bow_obj = CountVectorizer()
        bow_matrix = bow_obj.fit_transform(clean_corpus)

        # Store the term names in array.
        bow_terms = bow_obj.get_feature_names_out()
        
        return bow_matrix, bow_terms

    
    def tfidf_vectorize(self, clean_corpus):
        """
        Create a TF-IDF document-term matrix.

        Parameter:
            clean_corpus (list): Return value of method preprocess().

        Return values:
            tfidf_matrix (scipy.sparse._csr.csr_matrix): Token count document-term matrix.
            tfidf_terms (numpy.ndarray): Array of term names.
        """

        # Create a TF-IDF document-term matrix from the training data.
        tfidf_obj = TfidfVectorizer()
        tfidf_matrix = tfidf_obj.fit_transform(clean_corpus)

        # Store the term names in array.
        tfidf_terms = tfidf_obj.get_feature_names_out()
        
        return tfidf_matrix, tfidf_terms

    
    def plot_review_terms(self, matrix, terms, review_no):
        """
        Plot the 10 most relevant words of a review on a horizontal bar graph.

        Parameters:
            matrix (scipy.sparse._csr.csr_matrix): Return value of method bow_vectorize() or tfidf_vectorize().
            terms (numpy.ndarray): Return value of method bow_vectorize() or tfidf_vectorize().
            review_no (int): Topic number to extract most relevant terms.
        """

        # Create list of top words of topic.
        topic_vector = matrix.getrow(review_no-1).toarray()
        zipped = zip(topic_vector[0], terms)
        sorted_terms = sorted(zipped, reverse=True)
        terms_names = [i for v,i in sorted_terms[:10]]
        terms_values = [v for v,i in sorted_terms[:10]]
        
        # Plot as horizontal bar graph.
        fig, ax = plt.subplots(layout='constrained')
        ax.barh(terms_names, terms_values)
        ax.invert_yaxis()
        ax.set_title(f'Top Ten Words in Topic {review_no}')
        ax.set_xlabel('Value.')
        plt.show()

        return None

    
    def lsa_analysis(self, matrix, no_topics=19):
        """ 
        Create a Latent Semantic Analysis model and a by Singular Value Decomposition truncated matrix (array) from the original document-term matrix.
        
        Parameters:
            matrix (scipy.sparse._csr.csr_matrix): Return value of method bow_vectorize() or tfidf_vectorize().
            no_topics (int, opt.): The chosen number of topics for the model, default is 19 (best C_V value of best model).

        Return values:
            lsa_model (sklearn.decomposition._truncated_svd.TruncatedSVD): An LSA model with the chosen number of topics.
            lsa_array (numpy.ndarray): A truncated matrix (array) created from the LSA model and input matrix.
        """
        
        # Create an LSA model with the chosen number of topics and fit and transform it with the BoW or TF-IDF matrix. The random state is specified for reproducible results.
        lsa_model = TruncatedSVD(n_components=no_topics,random_state=42)
        lsa_array = lsa_model.fit_transform(matrix)
        
        return lsa_model, lsa_array


    def lda_analysis(self, matrix, no_topics=19):
        """ 
        Create a model and array according to Latent Dirichlet Allocation probability distributions from the original document-term matrix. 
        
        Parameters:
            matrix (scipy.sparse._csr.csr_matrix): Return value of method bow_vectorize() or tfidf_vectorize().
            no_topics (int, opt.): The chosen number of topics for the model, default is 19 (best C_V value of best model).

        Return values:
            lda_model (sklearn.decomposition._lda.LatentDirichletAllocation): An LDA model with the chosen number of topics.
            lda_array (numpy.ndarray): A truncated matrix (array) created from the LDA model and input matrix.
        """
        
        # Create an LDA model with the chosen number of topics and fit and transform it with the BoW or TF-IDF matrix. The random state is specified for reproducible results.
        lda_model = LatentDirichletAllocation(n_components=no_topics, random_state=42)
        lda_array = lda_model.fit_transform(matrix)

        return lda_model, lda_array



    def top_terms(self, model, terms, no_terms=20):
        """
        Extract the top terms per topic by relevance.

        Parameters:
            model (sklearn.decomposition._truncated_svd.TruncatedSVD OR sklearn.decomposition._lda.LatentDirichletAllocation): Return value of method lsa_analysis() or lda_analysis().
            terms (numpy.ndarray): Return value of method bow_vectorize() or tfidf_vectorize().
            no_terms (int, opt.): The specified number of topics, default is 20.

        Return value:
            top_terms (list): List of lists of top terms per topic. 
        """
        
        top_terms = []
        
        for i in enumerate(model.components_):
            # Combine document-term values with terms for parallel iteration.
            zipped = zip(i[1],terms, strict=True)
            top_terms_topic = []
            # Create descending list of terms with highest values for topics.
            for j in sorted(zipped, reverse=True)[:20]:
                top_terms_topic.append(j[1])
            top_terms.append(top_terms_topic)        
            
        return top_terms

       
    def get_coherence(self, top_terms, clean_corpus):
        """
        Calculate the C_V coherence score for a topic model.

        Parameters:
            top_terms (list): Return value of method top_terms().
            clean_corpus (list): Return value of method preprocess().

        Return value:
            coherence_score (numpy.float64): C_V value of specified topic model.
        """

        # Create list of tokenized reviews.
        texts = [word_tokenize(review) for review in clean_corpus]
        # Create a gensim dictionary mapping term to ID.
        dict = Dictionary(texts)
        # Create a coherence model for C_V and calculate the score.
        coherence_model = CoherenceModel(topics=top_terms, texts=texts, dictionary=dict, coherence='c_v') 
        coherence_score = coherence_model.get_coherence()
        
        return coherence_score

    
    def compare_coherence(self, mod_alg, matrix, terms, clean_corpus, max_topic=30):
        """ 
        Print the C_V coherence scores for a topic model with differing numbers of topics.

        This functions calls three other functions with a range of numbers of topics and prints the relavant C_V coherence scores, starting at 1 topic until a specified maximum of topics. 

        Parameters:
            mod_alg (str): The model algorithm to use, must be either 'lsa' or 'lda'.
            matrix (scipy.sparse._csr.csr_matrix): Return value of method bow_vectorize() or tfidf_vectorize().
            terms (numpy.ndarray): Return value of method bow_vectorize() or tfidf_vectorize().
            clean_corpus (list): Return value of method preprocess().
            max_topic (int, opt.): The maximum number of topics to calculate the coherence score for, default is 30. 

        Return value:
            None
        """

        # Different functions have to be called for LSA and LDA models.
        if mod_alg == 'lsa':
            print('\nLatent Semantic Analysis')
            for i in range(1, max_topic+1):
                # Create a model for i topics.
                _model = self.lsa_analysis(matrix, i)
                # Calculate the top terms for i topics.
                top_terms = self.top_terms(_model[0], terms)
                # Calculate and print the C_V coherence score for i topics.
                coherence_score = self.get_coherence(top_terms, clean_corpus)
                print(f'C_V value for {i} topics: ', coherence_score)
        elif mod_alg == 'lda':
            print('\nLatent Dirichlet Allocation')
            for i in range(1, max_topic+1):
                # Create a model for i topics.             
                _model = self.lda_analysis(matrix, i)
                # Calculate the top terms for i topics.
                top_terms = self.top_terms(_model[0], terms)
                # Calculate and print the C_V coherence score for i topics.
                coherence_score = self.get_coherence(top_terms, clean_corpus)
                print(f'C_V value for {i} topics: ', coherence_score)
        
        return None

    
    def plot_model(self, array, top_terms):
        """
        Represent the topic occurence probabilities of a model visually in a horizontal bar chart.

        Parameters:
            array (numpy.ndarray): Return value of method lsa_analysis() or lda_analysis().
            top_terms (list): Return value of method top_terms().

        Return value:
            None
        """

        # Transform top five terms into list of strings.
        top_five_terms = [', '.join(top_terms[i][:5]) for i,j in enumerate(top_terms)]
        
        # Create list of topic occurence likelihood.
        topic_occur = []
        # Iterate along first dimension of array.
        for i,m in enumerate(array[0]):
            x = 0
            # Iterate along second dimension of array and add up positive values to list of topic occurence likelihood.
            for j,n in enumerate(array):
                if array[j][i]>0:
                    x += array[j][i]
            topic_occur.append(x)

        # Transform list of topic occurence likelihood to that of probability values.
        topic_occur = [n/sum(topic_occur) for n in topic_occur]

        # Plot as horizontal bar graph.
        fig, ax = plt.subplots(layout='constrained')
        ax.barh(top_five_terms, topic_occur)
        ax.set_title('Topic Occurences')
        ax.set_xlabel('Probability of topic occurence within reviews of the corpus.')
        plt.show()

        return None
