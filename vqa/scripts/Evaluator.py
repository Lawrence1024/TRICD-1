import json
import os
class Evaluator:
    name = ""
    result_dict = {}
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        return f"{self.name}"
    
    def evaulate(self, result_filename):
        cmd = "python ../evaluation/main.py --results_file " + result_filename
        os.system(cmd)
        