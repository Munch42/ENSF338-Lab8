class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = []  # Each edge is a tuple: (weight, node1, node2)

    def add_edge(self, node1, node2, weight):
        self.nodes.add(node1)
        self.nodes.add(node2)
        self.edges.append((weight, node1, node2))

    # UNION-FIND helper: find root of set with path compression
    def find(self, parent, node):
        if parent[node] != node:
            parent[node] = self.find(parent, parent[node])
        return parent[node]

    # UNION-FIND helper: union by rank
    def union(self, parent, rank, node1, node2):
        root1 = self.find(parent, node1)
        root2 = self.find(parent, node2)

        if root1 != root2:
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1

    # Kruskal's algorithm to return the MST as a new Graph object
    def mst(self):
        mst_graph = Graph()
        parent = {}
        rank = {}

        # Initialize disjoint sets
        for node in self.nodes:
            parent[node] = node
            rank[node] = 0

        # Sort edges by ascending weight
        sorted_edges = sorted(self.edges)

        for weight, u, v in sorted_edges:
            # If adding the edge doesn’t form a cycle
            if self.find(parent, u) != self.find(parent, v):
                mst_graph.add_edge(u, v, weight)
                self.union(parent, rank, u, v)

        return mst_graph

    # For debug/printing the graph
    def print_graph(self):
        for weight, u, v in self.edges:
            print(f"{u} -- {v} [weight = {weight}]")


if __name__ == "__main__":
    g = Graph()
    g.add_edge('A', 'B', 2)
    g.add_edge('A', 'C', 3)
    g.add_edge('B', 'C', 1)
    g.add_edge('B', 'D', 4)
    g.add_edge('B', 'E', 6)
    g.add_edge('C', 'F', 5)
    g.add_edge('F', 'G', 2)

    print("Original Graph:")
    g.print_graph()

    mst = g.mst()
    print("\nMinimum Spanning Tree:")
    mst.print_graph()

"""
Look at ex3.pdf for graph drawings!
"""