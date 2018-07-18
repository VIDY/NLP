
from __future__ import print_function

import models.category_model

import os
import json
import sys
import json

from flask import Flask, request
app = Flask(__name__)
app.debug = True

category_graph = models.category_model.CategoryModel(); print("Category Graph loaded")

@app.route('/analyze-category', methods=['POST'])
def analyze_category():
    input_data=request.json
    if len(input_data)<2:
      return '{"error":"at least 2 sentences required as input"}'

    result=category_graph.eval(input_data)
    return json.dumps(result)

