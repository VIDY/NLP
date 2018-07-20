
from __future__ import print_function


import models.offensive.offensive_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

offensive_graph = models.offensive.offensive_model.OffensiveModel(); print("Graph loaded")

@app.route('/analyze-offensive', methods=['POST'])
def analyze_category():
    input_data=request.json

    result=offensive_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'

