class Graph:
    def __init__(self):
        self.nodes = []

    def insertNodeAndPad(self, index, node):
        if index >= len(self.nodes):
            self.nodes.extend([None] * (index - len(self.nodes) + 1))
        self.nodes[index] = node

    def addNode(self, data, index=None):
        if index is None:
            # Find the first None slot or append at the end
            index = next((i for i, node in enumerate(self.nodes) if node is None), len(self.nodes))
        
        node = GraphNode(data, index)
        self.insertNodeAndPad(index, node)
        return node

    def removeNode(self, node):
        for connection in node.connections.copy():  # Use copy to avoid modification during iteration
            connection.removeConnection(node)
        
        self.nodes[node.index] = None

    def addEdge(self, n1, n2, weight):
        n1.addConnection(n2, weight)
        n2.addConnection(n1, weight)

    def removeEdge(self, n1, n2):
        n1.removeConnection(n2)
        n2.removeConnection(n1)

    def importFromFile(self, fileName):
        self.nodes = []

        try:
            with open(fileName, 'r') as file:
                content = file.read().strip()
                
                # Check if it starts with "strict graph" and has proper braces
                if not content.startswith('strict graph') or not content.find('{') > 0 or not content.endswith('}'):
                    return None
                
                # Extract the content between braces
                start_brace = content.find('{')
                end_brace = content.rfind('}')
                if start_brace < 0 or end_brace < 0:
                    return None
                    
                content = content[start_brace+1:end_brace].strip()
                
                # Process each edge definition
                # Get the data from each line and store it in edges
                edges = [edge.strip() for edge in content.split(';') if edge.strip()]
                
                # Dictionary to keep track of created nodes
                node_dict = {}
                
                for edge in edges:
                    # Split edge definition and attributes
                    if '[' in edge:
                        edge_def, attributes = edge.split('[', 1)
                        edge_def = edge_def.strip()
                        attributes = attributes.strip()
                        if not attributes.endswith(']'):
                            return None
                        attributes = attributes[:-1]  # Remove closing ]
                    else:
                        edge_def = edge.strip()
                        attributes = ""
                    
                    # Parse nodes and edge token
                    parts = edge_def.split()
                    if len(parts) != 3 or parts[1] != '--':
                        return None
                        
                    node1_name, node2_name = parts[0], parts[2]
                    
                    # Parse weight attribute
                    weight = 1  # Default weight if they do not specify a weight in attributes
                    if attributes:
                        attr_pairs = attributes.split(',')
                        for pair in attr_pairs:
                            key, value = pair.split('=', 1)
                            if key.strip() == 'weight':
                                try:
                                    weight = int(value.strip())
                                except ValueError:
                                    return None
                    
                    # Create nodes if they don't exist
                    if node1_name not in node_dict:
                        node_dict[node1_name] = self.addNode(node1_name)
                    if node2_name not in node_dict:
                        node_dict[node2_name] = self.addNode(node2_name)
                    
                    # Add edge
                    self.addEdge(node_dict[node1_name], node_dict[node2_name], weight)
                    
                return self
                
        except Exception as e:
            print(f"Error: {e}")
            return None


class GraphNode:
    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.connections = []
        self.weights = []

    def addConnection(self, connectedNode, weightToNode):
        if connectedNode not in self.connections:
            self.connections.append(connectedNode)
            self.weights.append(weightToNode)

    def removeConnection(self, connectedNode):
        if connectedNode in self.connections:
            i = self.connections.index(connectedNode)
            del self.connections[i]
            del self.weights[i]

test = Graph()
test.importFromFile("random.dot")
