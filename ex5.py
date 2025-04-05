#1 Topological sorting can be implemented using Depth First Search
# You keep searching until you find a leaf node, then pop it of the graph. 
#This is done recursively, so at the end of it, you get the result backwards, 
#which can then be reversed. 
# We use DFS because it works only on DAGs this ensures that every node appears after its predecessors

class Graph:
    def __init__(self):
        self.adj = {}  # key = node, value = set of neighbors

    def addNode(self, node):
        if node not in self.adj:
            self.adj[node] = set()

    def addEdge(self, src, dest):
        self.addNode(src)
        self.addNode(dest)
        self.adj[src].add(dest)

    def isdag(self):
        visited = set()
        rec_stack = set()

        def dfs(v):
            visited.add(v)
            rec_stack.add(v)

            for neighbor in self.adj.get(v, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(v)
            return False

        for node in self.adj:
            if node not in visited:
                if dfs(node):
                    return False
        return True

    def toposort(self):
        if not self.isdag():
            return None

        visited = set()
        result = []

        def dfs(v):
            visited.add(v)
            for neighbor in self.adj.get(v, []):
                if neighbor not in visited:
                    dfs(neighbor)
            result.append(v)

        for node in self.adj:
            if node not in visited:
                dfs(node)

        result.reverse()
        return result


# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.addEdge("A", "B")
    g.addEdge("B", "C")
    g.addEdge("A", "C")

    print("Is DAG:", g.isdag())
    print("Topological Sort:", g.toposort())
    