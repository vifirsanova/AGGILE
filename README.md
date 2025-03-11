# AGGILE: Automated Graph Generation for Inference and Language Exploration
An instrument for automated graph generation from unstructured data in a controllable and predictable way. Extracts key entities and concepts from the text and generates predicates between them. Based on LLM-prompting.
## Getting Started
You may find walkthrough and sample html-graph in examples folder. Sample json-output is in outputs folder.
Demo is accessible via [link](https://huggingface.co/spaces/missvector/AGGILE).

Import and sample usage of AGGILE:
```python
import sys
sys.path.append('/home/aggile')

from aggile import Aggile, Graph

# Initialize Aggile with your HuggingFace credentials; change the model if needed
aggile = Aggile(model='deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', token='YOUR TOKEN')
# Form triplets from the text
triplets = aggile.form_triples('This is a sample text')
# Visualize graph based on generated triplets
Graph(tripletspip).build_graph() # Saves graph_with_predicates.html
```
If you use bash:
```bash
source .venv/bin/activate
chmod -x example_usage.py
python3 example_usage.py
```
**Note** that you should upload texts **500-1000 characters** long, otherwise the system prompt of generating 10 keywords will be invalid.

Example of graph:
![graph](examples/graph.png)
## Using AGGILE for Graph-based RAG
You can apply AGGILE for Graph-based question-answering as follows:
```python
# Initialize LLM agent
aggile = Aggile(model='HuggingFaceH4/zephyr-7b-beta', token='YOUR TOKEN')
# Provide your question and supporting text for building a graph
text = '''There are several theories as to the origin of the name "pug" <...>'''
question = 'What is the etymology of the word "Pug"?'
# Apply `graph_rag` method to get the answer
answer = aggile.graph_rag(text, question)
```
## Citation
tbd
