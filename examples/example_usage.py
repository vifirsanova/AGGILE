import sys
sys.path.append('/home/aggile')

from aggile import Aggile, Graph

aggile = Aggile(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', token='YOUR TOKEN')
triplets = aggile.form_triples('This is a sample text')
Graph(triplets).build_graph()