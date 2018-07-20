
from __future__ import print_function


import models.offensive.sentiment_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

sentiment_graph = models.offensive.sentiment_model.OffensiveModel(); print("Graph loaded")

@app.route('/analyze-offensive', methods=['POST'])
def analyze_category():
    input_data=request.json

    result=sentiment_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'

