class NNGraph: 
    def __init__(self, layers): 
        self.paths = []
        self.graph = graph = {
            "edges_values": {},
            "edges": {},
            "roots": []
        }
        self.layers = layers

    def build_nn_graph(self):
        for layer_index, layer in enumerate(self.layers):
            W = layer.weights[0].numpy()
            b = layer.weights[1].numpy()

            prev_nodes_lenght = W.shape[0]
            current_nodes_lenght = W.shape[1]

            for i in range(prev_nodes_lenght):
                node1_name = "layer-{}-node-{}".format(layer_index, i)
                adjacency_node_1 = self.graph["edges"].get(node1_name, [])

                ## I'am in the input layer
                if layer_index == 0:
                    self.graph["roots"].append(node1_name)

                for j in range(current_nodes_lenght):
                    node2_name = "layer-{}-node-{}".format(layer_index+1, j)
                    edge_name = "{}->{}".format(node1_name, node2_name)

                    self.graph["edges_values"][edge_name] = W[i, j]    
                    adjacency_node_1.append(node2_name)
                    
                self.graph["edges"][node1_name] = adjacency_node_1

        return self.graph

    def compute_paths(self):
        roots = self.graph["roots"]

        for root in roots:
            self.set_paths(root, [[], 1])

        return self.paths

    #private
    def set_paths(self, root, cumulative):
        # cheguei em um nó de output, temos um caminho.
        if root not in self.graph["edges"]:
            self.paths.append(cumulative)
            return
        
        edges = self.graph["edges"][root]
        for edge in edges:
            edge_name = root+'->'+edge
            new_path = list(cumulative[0])
            new_path.append(edge_name)
            self.set_paths(edge, [new_path, cumulative[1] * self.graph["edges_values"][edge_name]])

