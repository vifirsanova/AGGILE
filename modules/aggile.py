# aggile.py
class Aggile:
    """
    Graph generator for plain text
    """
    def __init__(self, model, token):
        from huggingface_hub import InferenceClient

        self.model = model
        self.token = token
        self.client = InferenceClient(model=model, token=token)

        n = None
        self.subj_prompt = f"""
        extract {n} collocations descibing key concepts, keywords, named entities from the provided source
        """

        self.obj_prompt = """
        extract 5-10 most repesentative collocations from the provided source that are related to the provided concept
        """

        self.pred_prompt = """
        define the relationship between two words: generate a verb or a phrase decribing a relationship between two entities; return a predicate for a knowledge graph triplet
        """

    def _get_subj(self, text, n=10):
        """
        Extract entities from the text:
            - named entities
            - kewords
            - concepts

        :text: input text (str)
        :n: the number of genrated entities (int)

        :return: {core_concepts: list of extracted keywords (subjects that will form triplets)} (dict)
        """
        import ast
        # Generate keywords from the given text using LLM
        core_concepts = self.client.chat.completions.create(messages=
                                                            [
                                                                {
                                                                    "role": "system", 
                                                                    "content": self.subj_prompt
                                                                },
                                                                {
                                                                    "role": "user", 
                                                                    "content": text
                                                                },
                                                            ],
                                                            response_format=
                                                            {
                                                                "type": "json",
                                                                "value": 
                                                                {
                                                                    "properties": 
                                                                    {
                                                                        "core_concepts": 
                                                                        {
                                                                            "type": "array", 
                                                                            "items": 
                                                                            {
                                                                                "type": "string"
                                                                            }
                                                                        },
                                                                    }
                                                                }
                                                            },
                                                            stream=False,
                                                            max_tokens=1024,
                                                            temperature=0.5,
                                                            top_p=0.1
                                                            ).choices[0].get('message')['content']
        return ast.literal_eval(core_concepts)
    
    def __extract_relations(self, word, text):
        import ast
        """
        Extract relation for the provided concepts (subjects) based on the information from the text:
            - collocations

        :text: input text (str)
        :concepts: the list of kewords and other key concepts extracted with aggile._get_subj (dict)
        
        :return: {related_concepts: list of related words and collocations (objects that will form triplets)} (dict)
        """
        related_concepts = self.client.chat.completions.create(messages=
                                                               [
                                                                   {
                                                                       "role": "system", 
                                                                       "content": self.obj_prompt
                                                                    },
                                                                    {
                                                                        "role": "user", 
                                                                        "content": f"concept = {word}, source = {text}"
                                                                    },
                                                                ],
                                                                response_format=
                                                                {
                                                                    "type": "json",
                                                                    "value": 
                                                                    {
                                                                        "properties": 
                                                                        {
                                                                            "related_concepts": 
                                                                            {
                                                                                "type": "array", 
                                                                                "items": 
                                                                                {
                                                                                    "type": "string"
                                                                                }
                                                                            },
                                                                        }
                                                                    }
                                                                },
                                                                stream=False,
                                                                max_tokens=512,
                                                                temperature=0.5,
                                                                top_p=0.1
                                                                ).choices[0].get('message')['content']
        return ast.literal_eval(related_concepts)
        
    def _get_obj(self, text):
        """
        Execute the extraction of related concepts for the list of keywords: 
            - generate list of objects for each object in the dictionarytract relation for the provided concepts (subjects) based on the information from the text:

        :text: input text (str)
        :concepts: the list of kewords and other key concepts extracted with aggile._get_subj (dict)
            
        :return: {related_concepts: list of related words and collocations (objects that will form triplets)} (dict)
        """
        # Generate list of subjects
        core_concepts = self._get_subj(text, n=10)
        # Get object for each subject
        relations = {word: self.__extract_relations(word, text) for word in core_concepts['core_concepts']}
        return relations
    
    def __generate_predicates(self, subj, obj):
        import ast
        """
        Generate predicates between objects and subjects

        :subj: one generated subject from core_condepts (str)
        :obj: one generated object from relations (str) 
        :text: input text (str)
            
        :return: one relevant predicate to form triplets (str)
        """
        predicate = self.client.chat.completions.create(messages=
                                                        [
                                                            {
                                                                "role": "system", 
                                                                "content": self.pred_prompt
                                                            },
                                                            {
                                                                "role": "user", 
                                                                "content": f"what is the relationship between {subj} and {obj}? return a predicate only"
                                                            },
                                                        ],
                                                        response_format=
                                                        {
                                                            "type": "json",
                                                            "value": 
                                                            {
                                                                "properties": 
                                                                {
                                                                    "predicate": 
                                                                    {
                                                                        "type": "string"
                                                                    },
                                                                }
                                                            }
                                                        },
                                                        stream=False,
                                                        max_tokens=512,
                                                        temperature=0.5,
                                                        top_p=0.1
                                                        ).choices[0].get('message')['content']
        return ast.literal_eval(predicate)['predicate'] # Return predicate only, not the whole dictionary

    def form_triples(self, text):
        """
        :text: input text (str) if from_string=True
        """

        # Generate objects from text
        relations = self._get_obj(text)
        # Placeholder for triplets
        triplets = dict()
        # Form triplets for each subject
        for subj in relations:
            # Placeholder for the current subject
            triplets[subj] = list()
            # For each object generated for this subject:
            for obj in relations[subj]['related_concepts']:
                # Create placeholder with the triplet structure "subject-predicate-object"
                temp = {'subject': subj, 'predicate': '', 'object': ''}
                # Save the object to the triplet
                temp['object'] = obj
                # Generate predicate between the current object and the current subject
                temp['predicate'] = self.__generate_predicates(subj, obj)
                # Hallucincation check: if object and subjects are the same entities, do not append them to the list of triplets
                if temp['subject'] != temp['object']:
                    # Otherwise, append the triplet 
                    triplets[subj].append(temp)
        
        return triplets

class Graph:
    def __init__(self, triplets):
        self.triplets = triplets
    
    def build_graph(self):
        import plotly.graph_objects as go
        import networkx as nx
        from collections import Counter
        import random

        # Prepare nodes and edges
        nodes = set()
        edges = []

        # Extract noded and edges from the set of triplets
        for key, values in self.triplets.items():
            for rel in values:
                nodes.add(rel['subject'])
                nodes.add(rel['object'])
                edges.append((rel['subject'], rel['object'], rel['predicate']))

        # Create a networkx graph
        G = nx.Graph()

        # Add nodes and edges to the graph
        for edge in edges:
            G.add_edge(edge[0], edge[1], label=edge[2])

        # Generate positions for nodes using force-directed layout with more space
        pos = nx.spring_layout(G, seed=42)  # Increasing k for more spacing

        # Extract node and edge data for Plotly
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_labels = list(G.nodes())

        # Count connections
        node_degrees = Counter([node for edge in edges for node in edge[:2]])

        # Assign distinct colors for each predicate (use a set to avoid duplicates)
        unique_predicates = list(set([edge[2] for edge in edges]))
        predicate_colors = {predicate: f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},1)'
                            for predicate in unique_predicates}

        # Plotly data for edges
        edge_x = []
        edge_y = []

        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        # Create the figure
        fig = go.Figure()

        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='text',
            mode='lines'
        ))

        # Add nodes with uniform size and labels
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=25,  # Uniform node size for all nodes
                color=[node_degrees[node] for node in node_labels],
                #colorscale='Viridis',
                colorbar=dict(title='Connections')
            ),
            text=node_labels,
            hoverinfo='text',
            textposition='top center',
            textfont=dict(size=13, weight="bold")
        ))

        # Add predicate labels near the nodes with black text
        for edge in edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            predicate_label = edge[2]

            # Calculate the midpoint of the edge and add small offsets to create spacing
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2

            # Add the label near the midpoint of the edge with black text
            fig.add_trace(go.Scatter(
                x=[mid_x], y=[mid_y],
                mode='text',
                text=[predicate_label],
                textposition='middle center',
                showlegend=False,
                textfont=dict(size=10)
            ))

        # Update layout
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            title="Force-Directed Graph with Predicate Labels on Nodes"
        )

        # Save the figure as an HTML file
        fig.write_html("graph_with_predicates.html")