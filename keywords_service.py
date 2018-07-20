
from __future__ import print_function


import models6.keywords_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

keywords_graph = models6.keywords_model.KeywordsModel(); print("Category Graph loaded")

@app.route('/analyze-keywords', methods=['POST'])
def analyze_category():
    input_data=request.json

    result=keywords_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'

