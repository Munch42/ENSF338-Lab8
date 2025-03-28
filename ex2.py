import timeit
import matplotlib.pyplot as plt

"""
1. Two of the possible ways to implement this queue can be an array that is searched using linear search resulting in an O(n^2) complexity or a Min Heap which allows for much faster 
access to the minimum edge weight node. If you use this min heap, the complexity imporves to O((|E| + |V|) log|V|) or basically O(nlog(n))
"""

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
        
    def slowSP(self, node):
        toBeChecked = ArrayQueue()

        for nodeInGraph in self.nodes:
            nodeInGraph.currDist = 9999999999
            nodeInGraph.pred = None
            toBeChecked.enqueue(nodeInGraph)

        node.currDist = 0

        while not toBeChecked.isEmpty():
            minCurDistNode = None
            minCurDist = 999999999999

            for item in toBeChecked.queue:
                if item.currDist < minCurDist:
                    minCurDist = item.currDist
                    minCurDistNode = item

            toBeChecked.remove_item(minCurDistNode)

            for neighbour in minCurDistNode.connections:
                if neighbour not in toBeChecked.queue:
                    continue

                edgeIndex = minCurDistNode.connections.index(neighbour)
                edgeWeight = minCurDistNode.weights[edgeIndex]

                tempDistance = minCurDistNode.currDist + edgeWeight

                if tempDistance < neighbour.currDist:
                    neighbour.currDist = tempDistance
                    neighbour.pred = minCurDistNode

        currDistList = []
        predList= []
        for nodeInGraph in self.nodes:
            currDistList.append(nodeInGraph.currDist)
            predList.append(nodeInGraph.pred)

        return currDistList, predList

    def fastSP(self, node):
        toBeChecked = MinHeap()

        for nodeInGraph in self.nodes:
            nodeInGraph.currDist = 9999999999
            nodeInGraph.pred = None
            toBeChecked.enqueue(nodeInGraph)

        node.currDist = 0

        while not toBeChecked.isEmpty():
            # We simply dequeue since it is a min heap so it automatically gives
            # us the node with the smallest distance
            minCurDistNode = toBeChecked.dequeue()
            
            for neighbour in minCurDistNode.connections:
                if not toBeChecked.contains(neighbour):
                    continue

                edgeIndex = minCurDistNode.connections.index(neighbour)
                edgeWeight = minCurDistNode.weights[edgeIndex]

                tempDistance = minCurDistNode.currDist + edgeWeight

                if tempDistance < neighbour.currDist:
                    neighbour.currDist = tempDistance
                    neighbour.pred = minCurDistNode
                    # This updates the heap with the new distance for the neighbour node so that it accurately gives the
                    # lowest currDist node for the following iterations
                    toBeChecked.decrease_key(neighbour)

        currDistList = []
        predList= []
        for nodeInGraph in self.nodes:
            currDistList.append(nodeInGraph.currDist)
            predList.append(nodeInGraph.pred)

        return currDistList, predList

class GraphNode:
    def __init__(self, data, index):
        self.data = data
        self.index = index
        self.connections = []
        self.weights = []

        self.currDist = 9999999999
        self.pred = None

    def addConnection(self, connectedNode, weightToNode):
        if connectedNode not in self.connections:
            self.connections.append(connectedNode)
            self.weights.append(weightToNode)

    def removeConnection(self, connectedNode):
        if connectedNode in self.connections:
            i = self.connections.index(connectedNode)
            del self.connections[i]
            del self.weights[i]

# MinHeap from lab 6 modified to support GraphNode with ChatGPT
class MinHeap:
    def __init__(self):
        self.heap = []
        self.pos_map = {}  # Maps nodes to their positions in the heap

    def __len__(self):
        return len(self.heap)

    def isEmpty(self):
        return len(self.heap) == 0

    def heapify(self, index):
        """Heapifies downwards from a given index to maintain heap property."""
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2

        if left < len(self.heap) and self.heap[left].currDist < self.heap[smallest].currDist:
            smallest = left

        if right < len(self.heap) and self.heap[right].currDist < self.heap[smallest].currDist:
            smallest = right

        if smallest != index:
            # Update position map when swapping
            self.pos_map[self.heap[index]] = smallest
            self.pos_map[self.heap[smallest]] = index
            # Swap elements
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self.heapify(smallest)

    def build_heap(self, arr):
        """Converts an input array into a valid heap."""
        self.heap = arr[:]  # Copy input array
        self.pos_map = {node: idx for idx, node in enumerate(arr)}
        for i in range(len(self.heap) // 2 - 1, -1, -1):
            self.heapify(i)  # Fix the heap property from bottom-up

    def enqueue(self, node):
        """Adds a node while maintaining heap properties."""
        self.heap.append(node)
        index = len(self.heap) - 1
        self.pos_map[node] = index
        parent = (index - 1) // 2

        while index > 0 and self.heap[parent].currDist > self.heap[index].currDist:
            # Update position map when swapping
            self.pos_map[self.heap[parent]] = index
            self.pos_map[self.heap[index]] = parent
            # Swap elements
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent
            parent = (index - 1) // 2

    def dequeue(self):
        """Removes and returns the node with smallest distance (min-heap root)."""
        if len(self.heap) == 0:
            return None

        root = self.heap[0]
        last_node = self.heap[-1]
        
        # Update position map
        del self.pos_map[root]
        
        if len(self.heap) > 1:
            self.pos_map[last_node] = 0
            self.heap[0] = last_node
            self.heap.pop()
            self.heapify(0)
        else:
            self.heap.pop()
            
        return root

    def decrease_key(self, node):
        """Update position of node after its priority (currDist) has been decreased."""
        if node not in self.pos_map:
            return
            
        index = self.pos_map[node]
        parent = (index - 1) // 2
        
        # If parent has higher distance, bubble up
        while index > 0 and self.heap[parent].currDist > self.heap[index].currDist:
            # Update position map when swapping
            self.pos_map[self.heap[parent]] = index
            self.pos_map[self.heap[index]] = parent
            # Swap elements
            self.heap[parent], self.heap[index] = self.heap[index], self.heap[parent]
            index = parent
            parent = (index - 1) // 2

    def contains(self, node):
        """Check if node is in the heap."""
        return node in self.pos_map

class ArrayQueue:
    def __init__(self):
        self.queue = []
    
    def enqueue(self, value):
        self.queue.insert(0, value)  # Insert at the head
    
    def dequeue(self):
        return self.queue.pop() if self.queue else None  # Remove from tail
    
    def remove_item(self, item):
        return self.queue.remove(item) if self.queue else None  # Remove a specific element to allow a linear search for Djikstra's
    
    def isEmpty(self):
        if not self.queue:
            return True
        else:
            return False
            
testGraph = Graph()
testGraph.importFromFile("random.dot")

#currDistsSlow, predsSlow = testGraph.slowSP(testGraph.nodes[0])
#currDistsFast, predsFast = testGraph.fastSP(testGraph.nodes[0])

slowTimes = []
fastTimes = []
for node in testGraph.nodes:
    testRuns = 5
    slowTime = timeit.timeit(lambda: testGraph.slowSP(node), number=testRuns) / testRuns
    fastTime = timeit.timeit(lambda: testGraph.fastSP(node), number=testRuns) / testRuns

    slowTimes.append(slowTime)
    fastTimes.append(fastTime)

avgSlowTime = sum(slowTimes) / len(slowTimes)
maxSlowTime = max(slowTimes)
minSlowTime = min(slowTimes)

avgFastTime = sum(fastTimes) / len(fastTimes)
maxFastTime = max(fastTimes)
minFastTime = min(fastTimes)

print("SlowSP Times:")
print("Average:", avgSlowTime)
print("Max:", maxSlowTime)
print("Min:", minSlowTime)

print("FastSP Times:")
print("Average:", avgFastTime)
print("Max:", maxFastTime)
print("Min:", minFastTime)

# Plotted with help from ChatGPT
# Replace the separate plots with this combined subplot approach
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

# First subplot for slowSP
ax1.hist(slowTimes, color='red', alpha=0.7)
ax1.set_title("Execution Times for slowSP")
ax1.set_xlabel("Time (seconds)")
ax1.set_ylabel("Frequency")
ax1.axvline(avgSlowTime, color='black', linestyle='dashed', linewidth=1, label=f'Avg: {avgSlowTime:.6f}s')
ax1.legend()

# Second subplot for fastSP
ax2.hist(fastTimes, color='blue', alpha=0.7)
ax2.set_title("Execution Times for fastSP")
ax2.set_xlabel("Time (seconds)")
ax2.set_ylabel("Frequency")
ax2.axvline(avgFastTime, color='black', linestyle='dashed', linewidth=1, label=f'Avg: {avgFastTime:.6f}s')
ax2.legend()

# Add overall title and adjust layout
fig.suptitle("Comparison of Execution Times: slowSP vs fastSP", fontsize=16)
plt.tight_layout()
fig.subplots_adjust(top=0.88)  # Make room for the overall title

plt.show()

# Debug code to print out the results of Dijkstra's 
"""
index = 1
for item in preds:
    if item == None:
        print("Source Node")
        continue

    print("\nNode")
    print(testGraph.nodes[index].data)
    print("Predecessor")
    print(item.data)
    print("Distance to Source:", currDists[index])
    index += 1
"""