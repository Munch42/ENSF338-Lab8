import time

class Graph2:
    def __init__(self):
        self.adj_matrix = []
        self.node_indices = {}
        self.index_nodes = {}
        self.size = 0

    def addNode(self, data):
        if data not in self.node_indices:
            self.node_indices[data] = self.size
            self.index_nodes[self.size] = data
            self.size += 1
            for row in self.adj_matrix:
                row.append(0)
            self.adj_matrix.append([0] * self.size)

    def addEdge(self, node1, node2, weight):
        i, j = self.node_indices[node1], self.node_indices[node2]
        self.adj_matrix[i][j] = weight
        self.adj_matrix[j][i] = weight

    def importFromFile(self, fileName):
        try:
            with open(fileName, 'r') as file:
                content = file.read().strip()
                start_brace = content.find('{')
                end_brace = content.rfind('}')
                content = content[start_brace + 1:end_brace].strip()
                edges = [edge.strip() for edge in content.split(';') if edge.strip()]
                for edge in edges:
                    if '[' in edge:
                        edge_def, attributes = edge.split('[', 1)
                        edge_def = edge_def.strip()
                        attributes = attributes.strip(']')
                    else:
                        edge_def, attributes = edge.strip(), ""

                    parts = edge_def.split()
                    if len(parts) != 3 or parts[1] != '--':
                        continue

                    node1, node2 = parts[0], parts[2]
                    weight = 1
                    if attributes:
                        for pair in attributes.split(','):
                            key, value = pair.split('=')
                            if key.strip() == 'weight':
                                weight = int(value.strip())

                    self.addNode(node1)
                    self.addNode(node2)
                    self.addEdge(node1, node2, weight)
        except Exception as e:
            print(f"Error: {e}")

    def (self):
        visited = [False] * self.size
        result = []

        def dfs_visit(v):
            visited[v] = True
            result.append(self.index_nodes[v])
            for u, connected in enumerate(self.adj_matrix[v]):
                if connected and not visited[u]:
                    dfs_visit(u)

        for v in range(self.size):
            if not visited[v]:
                dfs_visit(v)

        return result

def dfs_graph(graph):
    visited = set()
    result = []

    def dfs_visit(node):
        visited.add(node)
        result.append(node.data)
        for neighbor in node.connections:
            if neighbor not in visited:
                dfs_visit(neighbor)

    for node in graph.nodes:
        if node and node not in visited:
            dfs_visit(node)

    return result

# Import original Graph class and GraphNode from Exercise 1 here
from ex1 import Graph

def benchmark():
    graph1 = Graph()
    graph1.importFromFile("random.dot")
    g2 = Graph2()
    g2.importFromFile("random.dot")

    times1 = []
    for _ in range(10):
        start = time.perf_counter()
        dfs_graph(graph1)
        end = time.perf_counter()
        times1.append(end - start)

    times2 = []
    for _ in range(10):
        start = time.perf_counter()
        g2.dfs()
        end = time.perf_counter()
        times2.append(end - start)

    print("DFS on adjacency list:")
    print(f"Max: {max(times1):.6f}s, Min: {min(times1):.6f}s, Avg: {sum(times1)/len(times1):.6f}s")

    print("\nDFS on adjacency matrix:")
    print(f"Max: {max(times2):.6f}s, Min: {min(times2):.6f}s, Avg: {sum(times2)/len(times2):.6f}s")


if __name__ == '__main__':
    benchmark()

    # Discussion of results
    """
    Based on the benchmark, the adjacency list implementation is significantly faster.
    This is because:
    - It only stores edges that exist, reducing the memory footprint
    - Iterating over neighbors is O(degree), not O(n)
    The adjacency matrix always checks all nodes, which is slower especially for sparse graphs.
    
    Sample Output:
    
    DFS on adjacency list:
    Max: 0.005323s, Min: 0.000631s, Avg: 0.001216s

    DFS on adjacency matrix:
    Max: 0.056496s, Min: 0.034193s, Avg: 0.041912s
    
    """

