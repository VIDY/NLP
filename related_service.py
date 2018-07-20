
from __future__ import print_function
import models.related.related_model
import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

related_graph = models.related.related_model.RelatedModel()

@app.route('/analyze-related', methods=['POST'])
def analyze_related():
    input_data=request.json

    result=related_graph.eval(input_data)
    return json.dumps(result)
    #return 'Hello, World!'
