
from __future__ import print_function


import models.sentiment.sentiment_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

sentiment_graph = models.sentiment.sentiment_model.SentimentModel(); print("Category Graph loaded")

@app.route('/analyze-sentiment', methods=['POST'])
def analyze_category():
    input_data=request.json

    result=sentiment_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'

