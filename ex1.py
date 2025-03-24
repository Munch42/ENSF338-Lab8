class Graph:
    def __init__ (self):
        self.nodes = []

    def insertNodeAndPad(self, index, node):
        if index >= len(self.nodes):
            self.nodes.extend([None] * (index - len(self.nodes) + 1))
        self.nodes[index] = node

    def addNode(self, data):
        if self.nodes[len(self.nodes) - 1] == None:
            return self.addNode(data, len(self.nodes) - 1)
        else:
            return self.addNode(data, len(self.nodes))
    
    def addNode(self, data, index):
        node = GraphNode(data, len(self.nodes), index)
        self.insertNodeAndPad(index, node)
        return node

    def removeNode(self, node):
        for connection in node.connections:
            connection.removeConnection(node)
        
        self.nodes[node.index] = None

    def addEdge(self, n1, n2, weight):
        n1.addConnection(n2, weight)
        n2.addConnection(n1, weight)

    def removeEdge(self, n1, n2):
        n1.removeConnection(n2)
        n2.removeConnection(n1)


class GraphNode:
    def __init__ (self, data, index):
        self.data = data
        self.index = index
        self.connections = []
        self.weights = []

    def addConnection(self, connectedNode, weightToNode):
        self.connections.append(connectedNode)
        self.weights.append(weightToNode)

    def removeConnection(self, connectedNode):
        i = self.connections.index(connectedNode)
        del self.connections[i]
        del self.weights[i]
