import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
from fractions import Fraction

# Ensure the images directory exists
docs_dir = "docs"
img_dir = os.path.join(docs_dir, "1 Physics", "5 Circuits", "images")
os.makedirs(img_dir, exist_ok=True)

class EquivalentResistanceCalculator:
    def __init__(self, graph=None):
        """Initialize with an optional NetworkX graph."""
        self.graph = graph if graph is not None else nx.Graph()
        
    def create_graph_from_components(self, components):
        """
        Create a graph from a list of resistor components.
        
        Parameters:
        components -- list of tuples (node1, node2, resistance)
        """
        G = nx.Graph()
        for node1, node2, resistance in components:
            if G.has_edge(node1, node2):
                # If edge already exists, calculate parallel resistance
                current_resistance = G[node1][node2]['resistance']
                equivalent = 1 / (1/current_resistance + 1/resistance)
                G[node1][node2]['resistance'] = equivalent
            else:
                # Add new edge with resistance
                G.add_edge(node1, node2, resistance=resistance)
        
        self.graph = G
        return G
    
    def visualize_circuit(self, title="Circuit Graph", figsize=(10, 8), filename=None):
        """Visualize the circuit graph with resistance labels."""
        plt.figure(figsize=figsize)
        
        # Create a copy of the graph for visualization
        G = self.graph.copy()
        
        # Set up positions for nodes using spring layout
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
        
        # Draw edges with resistance labels
        edge_labels = {(u, v): f"{G[u][v]['resistance']:.2f} Ω" 
                     for u, v in G.edges()}
        
        nx.draw_networkx_edges(G, pos, width=2, alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
        
        # Draw node labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        
        plt.title(title)
        plt.axis('off')
        
        # Save the visualization to an image file
        if filename is None:
            filename = title.lower().replace(" ", "_")
        output_path = os.path.join(img_dir, f'{filename}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
        plt.close()
        
        return output_path
    
    def is_series_node(self, node):
        """Check if a node can be eliminated through series reduction."""
        # Node must have exactly 2 connections and not be a terminal node
        return self.graph.degree(node) == 2
    
    def perform_series_reduction(self):
        """Perform a single series reduction if possible."""
        G = self.graph
        
        for node in list(G.nodes()):
            if self.is_series_node(node):
                # Get the two neighbors
                neighbors = list(G.neighbors(node))
                n1, n2 = neighbors[0], neighbors[1]
                
                # Get the resistances
                r1 = G[n1][node]['resistance']
                r2 = G[node][n2]['resistance']
                
                # Remove the node
                G.remove_node(node)
                
                # Add direct connection between the neighbors if not already present
                if not G.has_edge(n1, n2):
                    G.add_edge(n1, n2, resistance=r1 + r2)
                else:
                    # If there's already a connection, calculate the equivalent
                    # (parallel combination of existing and new series)
                    existing_r = G[n1][n2]['resistance']
                    new_r = r1 + r2
                    G[n1][n2]['resistance'] = 1 / (1/existing_r + 1/new_r)
                
                return True, node, n1, n2  # Return info about the reduction
        
        return False, None, None, None  # No series reduction performed
    
    def perform_parallel_reduction(self):
        """Reduce parallel edges in the graph."""
        G = self.graph
        reduced = False
        
        # Find all multi-edges
        for node1 in G.nodes():
            for node2 in list(G.neighbors(node1)):
                if node1 < node2:  # Process each pair only once
                    # Check if the edge is a multi-edge
                    if isinstance(G[node1][node2], dict) and len(G[node1][node2]) > 1:
                        # Calculate equivalent resistance
                        resistances = [data['resistance'] for data in G[node1][node2].values()]
                        inv_sum = sum(1/r for r in resistances)
                        equivalent = 1 / inv_sum
                        
                        # Replace with a single edge
                        G.remove_edge(node1, node2)
                        G.add_edge(node1, node2, resistance=equivalent)
                        reduced = True
        
        return reduced
    
    def calculate_equivalent_resistance(self, source, target):
        """
        Calculate the equivalent resistance between source and target nodes.
        
        Returns the equivalent resistance or raises an exception if the nodes are not connected.
        """
        if source not in self.graph or target not in self.graph:
            raise ValueError("Source or target node not in graph")
        
        # If source and target are the same node, resistance is 0
        if source == target:
            return 0
        
        # Create a copy of the graph to work with
        self.working_graph = self.graph.copy()
        G = self.working_graph
        
        # Perform reductions until only source and target remain, or no more reductions possible
        reduction_performed = True
        while len(G.nodes()) > 2 and reduction_performed:
            reduction_performed = False
            
            # Try series reduction first
            for node in list(G.nodes()):
                if node not in [source, target] and G.degree(node) == 2:
                    # Get the two neighbors
                    neighbors = list(G.neighbors(node))
                    n1, n2 = neighbors[0], neighbors[1]
                    
                    # Get the resistances
                    r1 = G[n1][node]['resistance']
                    r2 = G[node][n2]['resistance']
                    
                    # Remove the node
                    G.remove_node(node)
                    
                    # Add direct connection between the neighbors or update existing one
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, resistance=r1 + r2)
                    else:
                        # If there's already a connection, it's a parallel combination
                        existing_r = G[n1][n2]['resistance']
                        new_r = r1 + r2
                        parallel_r = 1 / (1/existing_r + 1/new_r)
                        G[n1][n2]['resistance'] = parallel_r
                    
                    reduction_performed = True
                    break
            
            # If no series reduction, try parallel reduction
            if not reduction_performed:
                # Find pairs of nodes with multiple connections
                for n1 in G.nodes():
                    for n2 in G.nodes():
                        if n1 < n2 and G.has_edge(n1, n2):
                            # Check for multiple edges (in MultiGraph) - not applicable in simple Graph
                            # If we had a MultiGraph, we would handle parallel edges here
                            pass
        
        # Check if source and target are directly connected
        if G.has_edge(source, target):
            return G[source][target]['resistance']
        else:
            # If not, we need to solve the circuit using mesh/nodal analysis
            # For simplicity in this implementation, we'll just report that the circuit
            # requires advanced techniques
            raise ValueError("Could not reduce graph to single equivalent resistance. Advanced techniques required.")

    def delta_to_y_transformation(self, nodes):
        """
        Perform Delta-to-Y transformation on three nodes.
        
        Parameters:
        nodes -- a list of 3 nodes forming a triangle (delta)
        """
        G = self.graph
        if len(nodes) != 3 or not all(G.has_edge(nodes[i], nodes[j]) 
                                     for i in range(3) for j in range(i+1, 3)):
            raise ValueError("The three nodes must form a triangle in the graph")
        
        # Get the three resistances in the delta
        r12 = G[nodes[0]][nodes[1]]['resistance']
        r23 = G[nodes[1]][nodes[2]]['resistance']
        r31 = G[nodes[2]][nodes[0]]['resistance']
        
        # Calculate denominator for Y resistances
        denom = r12 + r23 + r31
        
        # Calculate Y resistances
        ra = (r12 * r31) / denom
        rb = (r12 * r23) / denom
        rc = (r23 * r31) / denom
        
        # Create a new node for the center of the Y
        center_node = "Y_center"
        while center_node in G.nodes():
            center_node += "_"
        
        # Remove the delta edges
        G.remove_edge(nodes[0], nodes[1])
        G.remove_edge(nodes[1], nodes[2])
        G.remove_edge(nodes[2], nodes[0])
        
        # Add the Y configuration
        G.add_node(center_node)
        G.add_edge(nodes[0], center_node, resistance=ra)
        G.add_edge(nodes[1], center_node, resistance=rb)
        G.add_edge(nodes[2], center_node, resistance=rc)
        
        return center_node

    def find_triangle(self):
        """Find a triangle (delta) in the graph."""
        G = self.graph
        
        # Look for 3-cycles in the graph
        for node1 in G.nodes():
            for node2 in G.neighbors(node1):
                for node3 in G.neighbors(node2):
                    if node3 != node1 and G.has_edge(node3, node1):
                        return [node1, node2, node3]
        
        return None

    def solve_complex_circuit(self, source, target):
        """
        Solve a complex circuit by applying delta-Y transformations when necessary.
        """
        # Try simple series-parallel reductions first
        try:
            result = self.calculate_equivalent_resistance(source, target)
            return result
        except ValueError:
            # If simple reductions don't work, try delta-Y transformations
            triangle = self.find_triangle()
            if triangle:
                self.delta_to_y_transformation(triangle)
                # Try again with the transformed graph
                return self.solve_complex_circuit(source, target)
            else:
                # If no triangle found, use a more general method like mesh analysis
                # This would be a more complex implementation beyond the scope of this solution
                raise ValueError("Circuit too complex for series-parallel and delta-Y reductions")

def format_resistance(value):
    """Format resistance value for display."""
    if isinstance(value, str):
        return value  # Already a formatted string
        
    # For simple integer values
    if value == int(value):
        return f"{int(value)} Ω"
    
    # Try to convert to fraction for cleaner representation
    try:
        f = Fraction(value).limit_denominator(1000)
        if f.denominator == 1:
            return f"{f.numerator} Ω"
        else:
            return f"{f.numerator}/{f.denominator} Ω"
    except:
        # Fallback to decimal representation
        return f"{value:.3f} Ω"

# Example 1: Simple series circuit
def example_series_circuit():
    calculator = EquivalentResistanceCalculator()
    
    # Create a simple series circuit: A--5Ω--B--3Ω--C
    components = [
        ('A', 'B', 5),  # 5 Ω resistor between A and B
        ('B', 'C', 3)   # 3 Ω resistor between B and C
    ]
    
    calculator.create_graph_from_components(components)
    calculator.visualize_circuit(title="Series Circuit", filename="circuit_graph_series")
    
    # Calculate equivalent resistance
    equiv_resistance = calculator.calculate_equivalent_resistance('A', 'C')
    
    result = {
        "circuit_type": "Series Circuit",
        "components": components,
        "equivalent_resistance": equiv_resistance,
        "explanation": "In a series circuit, the equivalent resistance is the sum of individual resistances."
    }
    
    return result

# Example 2: Parallel circuit
def example_parallel_circuit():
    calculator = EquivalentResistanceCalculator()
    
    # Create a parallel circuit: A connected to B by three resistors in parallel
    components = [
        ('A', 'B', 6),   # 6 Ω resistor between A and B
        ('A', 'B', 12),  # 12 Ω resistor between A and B
        ('A', 'B', 4)    # 4 Ω resistor between A and B
    ]
    
    # Since networkx Graph doesn't support parallel edges directly, we'll calculate the 
    # equivalent resistance manually for visualization
    equiv_parallel = 1 / (1/6 + 1/12 + 1/4)
    visualization_components = [('A', 'B', equiv_parallel)]
    
    # Create the circuit for visualization
    viz_calculator = EquivalentResistanceCalculator()
    viz_calculator.create_graph_from_components(visualization_components)
    viz_calculator.visualize_circuit(title="Parallel Circuit", filename="circuit_graph_parallel")
    
    # For calculation, we'll handle manually since our Graph implementation
    # doesn't directly support parallel edges
    equiv_resistance = equiv_parallel
    
    result = {
        "circuit_type": "Parallel Circuit",
        "components": components,
        "equivalent_resistance": equiv_resistance,
        "explanation": "In a parallel circuit, the equivalent resistance is calculated as 1/Req = 1/R1 + 1/R2 + 1/R3 + ..."
    }
    
    return result

# Example 3: Bridge circuit (Wheatstone bridge)
def example_bridge_circuit():
    calculator = EquivalentResistanceCalculator()
    
    # Create a bridge circuit (Wheatstone bridge)
    components = [
        ('A', 'B', 5),  # 5 Ω resistor
        ('A', 'D', 10), # 10 Ω resistor
        ('B', 'C', 20), # 20 Ω resistor
        ('B', 'E', 10), # 10 Ω resistor
        ('C', 'E', 5),  # 5 Ω resistor
        ('D', 'E', 20), # 20 Ω resistor
        ('D', 'C', 10)  # 10 Ω resistor
    ]
    
    calculator.create_graph_from_components(components)
    calculator.visualize_circuit(title="Bridge Circuit", filename="circuit_graph_bridge")
    
    # This is a complex circuit that may require delta-Y transformation
    try:
        equiv_resistance = calculator.solve_complex_circuit('A', 'C')
        explanation = "This bridge circuit was solved using a combination of series-parallel reduction and delta-Y transformations."
    except ValueError as e:
        equiv_resistance = "Complex - requires mesh analysis"
        explanation = f"This bridge circuit is too complex for simple reductions: {str(e)}"
    
    result = {
        "circuit_type": "Bridge Circuit (Wheatstone Bridge)",
        "components": components,
        "equivalent_resistance": equiv_resistance,
        "explanation": explanation
    }
    
    return result

# Run the examples
if __name__ == "__main__":
    print("Generating Circuit Analysis Examples")
    print("=" * 50)
    
    examples = [
        example_series_circuit(),
        example_parallel_circuit(),
        example_bridge_circuit()
    ]
    
    print("\nResults Summary:")
    print("=" * 50)
    for i, example in enumerate(examples, 1):
        print(f"Example {i}: {example['circuit_type']}")
        print(f"Equivalent Resistance: {format_resistance(example['equivalent_resistance'])}")
        print(f"Explanation: {example['explanation']}")
        print("-" * 50)
    
    print("\nAll circuit images have been saved to the docs/1 Physics/5 Circuits/images directory.")
    print("You can reference these images in your markdown documentation.") 