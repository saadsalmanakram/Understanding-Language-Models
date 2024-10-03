'''
The foundation of the Transformers library is the `pipeline()` function, which seamlessly integrates a model with its required preprocessing and postprocessing steps. This allows us to input any text and receive a coherent response:
'''

# python 

from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for HuggingFace course my whole life.")

#output
# [{'label': 'POSITIVE', 'score': 0.9598047137260437}]


# For multiline sentence 

classifier(
    ["I've been waiting for a HuggingFace couse my whole life.", "I hate thi so much!"]

#Output
#[{'label': 'POSITIVE', 'score': 0.9598047137260437},
# {'label': 'NEGATIVE', 'score': 0.9994558095932007}]


