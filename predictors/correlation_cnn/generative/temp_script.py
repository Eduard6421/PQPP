import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.parse.stanford import StanfordParser
import os

nltk.download("averaged_perceptron_tagger")

os.environ["JAVA_HOME"] = "C:/Program Files/Java/jre-1.8/bin"

print(os.listdir("../../../../stanford-postagger/models/"))

path_to_jar = "../../../../stanford-postagger/stanford-postagger-4.2.0.jar"
path_to_model_jar = (
    "../../../../stanford-postagger/models/english-bidirectional-distsim.tagger"
)

parser = StanfordParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_model_jar)

# Define your sentence
sentence = "The quick brown fox jumps over the lazy dog"

# Tokenize and POS tag the sentence
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)

# Parse the sentence
parse_tree = list(parser.parse(tokens))

# Draw the syntactic tree
parse_tree[0].draw()
