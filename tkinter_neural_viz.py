"""
Tkinter Neural Network Visualizer
A high-performance, customizable neural network visualization tool
with smooth animations and modern styling.
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import threading
import queue
import time
import math
from collections import defaultdict


@dataclass
class Node:
    """Represents a neuron in the network"""
    id: str
    layer_id: str
    layer_name: str
    index: int
    x: float = 0.0
    y: float = 0.0
    radius: float = 15.0
    activation: float = 0.0
    bias: float = 0.0
    visible: bool = True


@dataclass
class Edge:
    """Represents a connection between neurons"""
    source_id: str
    target_id: str
    weight: float = 0.0
    gradient: float = 0.0
    visible: bool = True
    layer_index: int = 0
    source_idx: int = 0
    target_idx: int = 0


@dataclass
class Layer:
    """Represents a layer in the network"""
    id: str
    name: str
    layer_type: str
    size: int
    index: int
    x_position: float = 0.0
    nodes: List[Node] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


class PyTorchParser:
    """Parser for PyTorch models"""
    
    def extract_weights(self, model: Any) -> Dict[str, np.ndarray]:
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    def extract_activations(self, model: Any, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[name] = output.detach().cpu().numpy()
            return hook
        
        hooks = []
        for name, module in model.named_modules():
            if name != '':
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        import torch
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_data = torch.from_numpy(input_data).float()
            model(input_data)
        
        for hook in hooks:
            hook.remove()
        
        return activations


class TensorFlowParser:
    """Parser for TensorFlow/Keras models"""
    
    def extract_weights(self, model: Any) -> Dict[str, np.ndarray]:
        weights = {}
        for layer in model.layers:
            layer_weights = layer.get_weights()
            if layer_weights:
                for i, w in enumerate(layer_weights):
                    weights[f"{layer.name}_weight_{i}"] = w
        return weights
    
    def extract_activations(self, model: Any, input_data: np.ndarray) -> Dict[str, np.ndarray]:
        import tensorflow as tf
        
        activations = {}
        layer_outputs = [layer.output for layer in model.layers]
        activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
        
        outputs = activation_model.predict(input_data, verbose=0)
        
        for i, layer in enumerate(model.layers):
            activations[layer.name] = outputs[i]
        
        return activations


class NetworkCanvas(tk.Canvas):
    """Custom canvas for rendering the neural network"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(bg='#0a0e17', highlightthickness=0)
        
        # Node and edge storage
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.layers: List[Layer] = []
        
        # Canvas item IDs
        self.node_items: Dict[str, int] = {}
        self.edge_items: List[int] = []
        self.edge_to_canvas: Dict[int, int] = {}  # Map edge object id to canvas item
        self.layer_labels: Dict[str, int] = {}
        
        # Animation properties
        self.animation_speed = 0.15
        self.pulse_phase = 0.0
        
        # Color scheme
        self.colors = {
            'positive': '#00ff88',
            'negative': '#ff4757',
            'neutral': '#3742fa',
            'edge': '#2f3542',
            'edge_active': '#ffa502',
            'text': '#f1f2f6',
            'layer_bg': 'rgba(30, 39, 46, 0.5)'
        }
        
        # Performance tracking
        self.update_count = 0
        self.last_update_time = time.time()
        self.fps = 0
        
        # Zoom/Pan functionality
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.original_nodes = {}  # Store original positions
        self.is_selecting = False
        self.selection_start = None
        self.selection_rect = None
        self._mouse_moved = False
        
        # Bind mouse events for selection
        self.bind('<Button-1>', self._on_mouse_down)
        self.bind('<B1-Motion>', self._on_mouse_drag)
        self.bind('<ButtonRelease-1>', self._on_mouse_up)
        
        # Node selection
        self.selected_nodes: Dict[str, Node] = {}
        self.selected_node_items: Dict[str, int] = {}  # node_id -> canvas item for highlight
        self.on_node_selected: Optional[Callable] = None  # Callback for node selection
        
        # Start animation loop
        self._animate()
    
    def create_gradient_circle(self, x, y, radius, color_inner, color_outer):
        """Create a gradient-filled circle using multiple ovals"""
        items = []
        steps = 8
        for i in range(steps, 0, -1):
            r = radius * i / steps
            # Interpolate color
            ratio = i / steps
            
            # Parse colors
            r1, g1, b1 = self._hex_to_rgb(color_inner)
            r2, g2, b2 = self._hex_to_rgb(color_outer)
            
            # Interpolate
            rf = int(r1 + (r2 - r1) * ratio)
            gf = int(g1 + (g2 - g1) * ratio)
            bf = int(b1 + (b2 - b1) * ratio)
            
            color = f'#{rf:02x}{gf:02x}{bf:02x}'
            
            item = self.create_oval(
                x - r, y - r, x + r, y + r,
                fill=color, outline=''
            )
            items.append(item)
        
        return items
    
    def _hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def draw_network(self, layers: List[Layer], edges: List[Edge]):
        """Draw the entire network"""
        self.layers = layers
        self.edges = edges
        
        # Clear existing items
        self.delete('all')
        self.node_items.clear()
        self.edge_items.clear()
        self.layer_labels.clear()
        
        # Draw edges first (behind nodes)
        self._draw_edges()
        
        # Draw nodes
        self._draw_nodes()
        
        # Draw layer labels
        self._draw_layer_labels()
    
    def _draw_edges(self):
        """Draw connections between nodes with weight visualization"""
        self.edge_items = []  # List of canvas item IDs
        self.edge_to_canvas = {}  # Map edge object id to canvas item
        
        for edge in self.edges:
            source = self._find_node(edge.source_id)
            target = self._find_node(edge.target_id)
            
            if source and target:
                # Calculate color and width based on weight
                weight = edge.weight
                
                # Color: positive = green, negative = red, zero = gray
                if weight > 0:
                    intensity = min(abs(weight) * 2, 1.0)  # Scale up for visibility
                    color = f'#{int(0 * intensity):02x}{int(255 * intensity):02x}{int(136 * intensity):02x}'
                elif weight < 0:
                    intensity = min(abs(weight) * 2, 1.0)
                    color = f'#{int(255 * intensity):02x}{int(71 * intensity):02x}{int(87 * intensity):02x}'
                else:
                    color = self.colors['edge']
                
                # Width: thicker for larger weights (1-4 pixels)
                width = max(1, min(4, abs(weight) * 3 + 0.5))
                
                line = self.create_line(
                    source.x, source.y, target.x, target.y,
                    fill=color,
                    width=width,
                    smooth=True
                )
                
                # Store canvas item IDs
                self.edge_items.append(line)
                self.edge_to_canvas[id(edge)] = line
    
    def _draw_nodes(self):
        """Draw all nodes with click bindings"""
        for layer in self.layers:
            for node in layer.nodes:
                self.nodes[node.id] = node
                
                # Create gradient circle
                items = self.create_gradient_circle(
                    node.x, node.y, node.radius,
                    self.colors['neutral'],
                    '#1e272e'
                )
                
                # Store the outermost circle for updates
                self.node_items[node.id] = items[-1]
                
                # Add glow effect
                glow = self.create_oval(
                    node.x - node.radius - 5,
                    node.y - node.radius - 5,
                    node.x + node.radius + 5,
                    node.y + node.radius + 5,
                    outline=self.colors['neutral'],
                    width=2,
                    stipple='gray50'
                )
                
                # Create larger invisible clickable area for easier selection
                click_area = self.create_oval(
                    node.x - node.radius - 15,
                    node.y - node.radius - 15,
                    node.x + node.radius + 15,
                    node.y + node.radius + 15,
                    fill='', outline='', tags=f'clickable_{node.id}'
                )
                self.tag_bind(f'clickable_{node.id}', '<Button-1>', lambda e, n=node: self._on_node_click(e, n))
                
                # Also bind to visual items
                for item in items:
                    self.tag_bind(item, '<Button-1>', lambda e, n=node: self._on_node_click(e, n))
    
    def _on_node_click(self, event, node: Node):
        """Handle node click with multi-selection support"""
        # Check if Ctrl is held
        ctrl_held = (event.state & 0x4) != 0
        
        if not ctrl_held:
            # Clear previous selection if Ctrl not held
            self._clear_selection()
        
        # Toggle selection
        if node.id in self.selected_nodes:
            # Deselect
            del self.selected_nodes[node.id]
            self._remove_node_highlight(node.id)
        else:
            # Select
            self.selected_nodes[node.id] = node
            self._highlight_node(node.id)
        
        # Notify callback
        if self.on_node_selected:
            self.on_node_selected(list(self.selected_nodes.values()))
    
    def _highlight_node(self, node_id: str):
        """Add highlight ring around selected node"""
        if node_id not in self.nodes:
            return
        
        node = self.nodes[node_id]
        
        # Create highlight ring
        highlight = self.create_oval(
            node.x - node.radius - 8,
            node.y - node.radius - 8,
            node.x + node.radius + 8,
            node.y + node.radius + 8,
            outline='#ffa502',  # Orange highlight
            width=3,
            tags='highlight'
        )
        self.selected_node_items[node_id] = highlight
        self.tag_lower('highlight')  # Put highlight behind nodes
    
    def _remove_node_highlight(self, node_id: str):
        """Remove highlight from node"""
        if node_id in self.selected_node_items:
            self.delete(self.selected_node_items[node_id])
            del self.selected_node_items[node_id]
    
    def _clear_selection(self):
        """Clear all selected nodes"""
        for node_id in list(self.selected_node_items.keys()):
            self._remove_node_highlight(node_id)
        self.selected_nodes.clear()
        if self.on_node_selected:
            self.on_node_selected([])
    
    def _draw_layer_labels(self):
        """Draw layer names and info"""
        for layer in self.layers:
            label = self.create_text(
                layer.x_position, 40,
                text=f"{layer.name}\n{layer.layer_type}\n({layer.size} units)",
                fill=self.colors['text'],
                font=('Helvetica', 9),
                justify='center'
            )
            self.layer_labels[layer.id] = label
    
    def _find_node(self, node_id: str) -> Optional[Node]:
        """Find a node by ID"""
        for layer in self.layers:
            for node in layer.nodes:
                if node.id == node_id:
                    return node
        return None
    
    def update_activations(self, activations: Dict[str, np.ndarray]):
        """Update node colors based on activations"""
        for layer in self.layers:
            layer_name = layer.name
            if layer_name in activations:
                acts = activations[layer_name]
                if isinstance(acts, np.ndarray):
                    if len(acts.shape) > 1:
                        acts = acts[0]  # Take first sample
                
                for i, node in enumerate(layer.nodes):
                    if i < len(acts):
                        node.activation = float(acts[i])
                        self._update_node_color(node)
    
    def _update_node_color(self, node: Node):
        """Update a single node's color based on activation"""
        if node.id not in self.node_items:
            return
        
        # Normalize activation to [-1, 1]
        act = np.tanh(node.activation)
        
        # Determine color
        if act > 0:
            color = self.colors['positive']
            intensity = min(abs(act) + 0.3, 1.0)
        elif act < 0:
            color = self.colors['negative']
            intensity = min(abs(act) + 0.3, 1.0)
        else:
            color = self.colors['neutral']
            intensity = 0.5
        
        # Update node appearance
        item_id = self.node_items[node.id]
        
        # Create new gradient
        self.delete(item_id)
        items = self.create_gradient_circle(
            node.x, node.y,
            node.radius + abs(act) * 5,
            color,
            '#1e272e'
        )
        self.node_items[node.id] = items[-1]
        
        # Raise above edges
        for item in items:
            self.tag_raise(item)
    
    def update_weights(self, weights: Dict[str, np.ndarray]):
        """Update edge colors and widths based on weights"""
        # Convert weight dictionary to list in layer order
        weight_list = []
        for key in sorted(weights.keys()):
            if 'weight' in key:
                weight_list.append(weights[key])
        
        # Update edge weights from weight matrices
        for edge in self.edges:
            layer_idx = edge.layer_index - 1  # Adjust for layer indexing
            if layer_idx >= 0 and layer_idx < len(weight_list):
                weight_matrix = weight_list[layer_idx]
                if edge.target_idx < weight_matrix.shape[0] and edge.source_idx < weight_matrix.shape[1]:
                    edge.weight = float(weight_matrix[edge.target_idx, edge.source_idx])
            
            if id(edge) in self.edge_to_canvas:
                canvas_item = self.edge_to_canvas[id(edge)]
                weight = edge.weight
                
                # Calculate color based on weight
                if weight > 0:
                    intensity = min(abs(weight) * 2, 1.0)
                    color = f'#{int(0 * intensity):02x}{int(255 * intensity):02x}{int(136 * intensity):02x}'
                elif weight < 0:
                    intensity = min(abs(weight) * 2, 1.0)
                    color = f'#{int(255 * intensity):02x}{int(71 * intensity):02x}{int(87 * intensity):02x}'
                else:
                    color = self.colors['edge']
                
                # Calculate width based on weight magnitude
                width = max(1, min(4, abs(weight) * 3 + 0.5))
                
                # Update the line
                self.itemconfig(canvas_item, fill=color, width=width)
    
    def _animate(self):
        """Animation loop for pulsing effects"""
        self.pulse_phase += self.animation_speed
        
        # Animate active nodes
        for node in self.nodes.values():
            if abs(node.activation) > 0.1:
                pulse = math.sin(self.pulse_phase + node.index * 0.5) * 0.1 + 1.0
                radius = node.radius * pulse
                
                if node.id in self.node_items:
                    item_id = self.node_items[node.id]
                    coords = self.coords(item_id)
                    if coords:
                        cx = (coords[0] + coords[2]) / 2
                        cy = (coords[1] + coords[3]) / 2
                        self.coords(item_id, cx - radius, cy - radius, cx + radius, cy + radius)
        
        # Update FPS counter
        self.update_count += 1
        current_time = time.time()
        if current_time - self.last_update_time >= 1.0:
            self.fps = self.update_count
            self.update_count = 0
            self.last_update_time = current_time
        
        self.after(50, self._animate)  # 20 FPS animation
    
    def get_fps(self) -> int:
        """Get current FPS"""
        return self.fps
    
    def _on_mouse_down(self, event):
        """Start selection or check for node click"""
        self.is_selecting = True
        self.selection_start = (event.x, event.y)
        self._mouse_moved = False
        
        # Check if clicked on a node using coordinate-based detection
        clicked_node = self._find_node_at_position(event.x, event.y)
        if clicked_node:
            # Simulate node click
            self._on_node_click(event, clicked_node)
            self.is_selecting = False
            return
        
        # Create selection rectangle for zoom selection
        if self.selection_rect:
            self.delete(self.selection_rect)
        self.selection_rect = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline='#00ff88', width=2, dash=(5, 5),
            fill='', tags='selection'
        )
        self.tag_lower('selection')  # Put selection behind network
    
    def _find_node_at_position(self, x, y):
        """Find node at canvas coordinates"""
        for layer in self.layers:
            for node in layer.nodes:
                # Check if click is within node radius + margin
                hit_radius = node.radius + 20  # Larger hit area
                dx = x - node.x
                dy = y - node.y
                distance = (dx * dx + dy * dy) ** 0.5
                if distance <= hit_radius:
                    return node
        return None
    
    def _on_mouse_drag(self, event):
        """Update selection rectangle while dragging"""
        if self.is_selecting and self.selection_rect and self.selection_start is not None:
            self._mouse_moved = True
            self.coords(self.selection_rect, 
                       self.selection_start[0], self.selection_start[1],
                       event.x, event.y)
    
    def _on_mouse_up(self, event):
        """Finish selection and zoom to selected region"""
        if not self.is_selecting or self.selection_start is None:
            self.is_selecting = False
            if self.selection_rect:
                self.delete(self.selection_rect)
                self.selection_rect = None
            return
        
        self.is_selecting = False
        
        # Remove selection rectangle
        if self.selection_rect:
            self.delete(self.selection_rect)
            self.selection_rect = None
        
        # Only zoom if actually dragged (not just clicked)
        if not self._mouse_moved:
            return
        
        # Get selection bounds
        x1 = min(self.selection_start[0], event.x)
        y1 = min(self.selection_start[1], event.y)
        x2 = max(self.selection_start[0], event.x)
        y2 = max(self.selection_start[1], event.y)
        
        # Only zoom if selection is large enough
        if abs(x2 - x1) > 50 and abs(y2 - y1) > 50:
            self.zoom_to_region(x1, y1, x2, y2)
    
    def zoom_to_region(self, x1, y1, x2, y2):
        """Zoom to a specific region of the canvas"""
        if not self.layers:
            return
        
        # Store original positions if not already stored
        if not self.original_nodes:
            for layer in self.layers:
                for node in layer.nodes:
                    self.original_nodes[node.id] = (node.x, node.y)
        
        # Calculate zoom and pan
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        selection_width = x2 - x1
        selection_height = y2 - y1
        
        # Calculate zoom factor to fit selection in canvas
        zoom_x = canvas_width / selection_width
        zoom_y = canvas_height / selection_height
        self.zoom_level = min(zoom_x, zoom_y) * 0.9  # 90% to leave some margin
        
        # Calculate pan to center the selection
        selection_center_x = (x1 + x2) / 2
        selection_center_y = (y1 + y2) / 2
        
        self.pan_x = (canvas_width / 2) - (selection_center_x * self.zoom_level)
        self.pan_y = (canvas_height / 2) - (selection_center_y * self.zoom_level)
        
        # Apply transformation to all nodes
        self._apply_transform()
        
        # Redraw network
        self._redraw_with_transform()
    
    def reset_zoom(self):
        """Reset zoom and pan to default"""
        self.zoom_level = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Restore original positions
        if self.original_nodes:
            for layer in self.layers:
                for node in layer.nodes:
                    if node.id in self.original_nodes:
                        orig_x, orig_y = self.original_nodes[node.id]
                        node.x = orig_x
                        node.y = orig_y
            self.original_nodes = {}
        
        # Redraw network
        self.delete('all')
        self._draw_network_from_scratch()
    
    def _apply_transform(self):
        """Apply zoom and pan transformation to node positions"""
        for layer in self.layers:
            for node in layer.nodes:
                orig_x = self.original_nodes.get(node.id, (node.x, node.y))[0]
                orig_y = self.original_nodes.get(node.id, (node.x, node.y))[1]
                node.x = orig_x * self.zoom_level + self.pan_x
                node.y = orig_y * self.zoom_level + self.pan_y
    
    def _redraw_with_transform(self):
        """Redraw network with current transformation"""
        # Save selected node IDs and their activations
        selected_ids = list(self.selected_nodes.keys())
        node_activations = {node_id: self.nodes[node_id].activation for node_id in self.nodes}
        
        # Clear canvas
        self.delete('all')
        
        # Redraw everything
        self._draw_edges()
        self._draw_nodes()
        self._draw_layer_labels()
        
        # Reapply activation colors
        for node_id, activation in node_activations.items():
            if node_id in self.nodes:
                self.nodes[node_id].activation = activation
                self._update_node_color(self.nodes[node_id])
        
        # Reapply selection highlights
        for node_id in selected_ids:
            if node_id in self.nodes:
                self._highlight_node(node_id)
                self.selected_nodes[node_id] = self.nodes[node_id]
        
        # Notify callback
        if self.on_node_selected and selected_ids:
            self.on_node_selected(list(self.selected_nodes.values()))
    
    def _draw_network_from_scratch(self):
        """Redraw network from original data"""
        # Save selected node IDs and their activations
        selected_ids = list(self.selected_nodes.keys())
        node_activations = {node_id: self.nodes[node_id].activation for node_id in self.nodes}
        
        self.draw_network(self.layers, self.edges)
        
        # Reapply activation colors
        for node_id, activation in node_activations.items():
            if node_id in self.nodes:
                self.nodes[node_id].activation = activation
                self._update_node_color(self.nodes[node_id])
        
        # Reapply selection highlights
        for node_id in selected_ids:
            if node_id in self.nodes:
                self._highlight_node(node_id)
                self.selected_nodes[node_id] = self.nodes[node_id]
        
        # Notify callback
        if self.on_node_selected and selected_ids:
            self.on_node_selected(list(self.selected_nodes.values()))


class TkinterNeuralVisualizer:
    """Main visualizer class using Tkinter"""
    
    def __init__(self, width: int = 1400, height: int = 800, max_nodes_per_layer: int = 24):
        self.width = width
        self.height = height
        self.max_nodes_per_layer = max_nodes_per_layer
        
        self.model = None
        self.parser = None
        self.layers: List[Layer] = []
        self.edges: List[Edge] = []
        self.node_map: Dict[str, Node] = {}
        
        self.update_queue = queue.Queue()
        self.is_running = False
        self.update_thread = None
        
        # Statistics
        self.total_updates = 0
        self.start_time = None
        
        # Step-by-step mode
        self.step_mode = False
        self.step_count = 0
        self.waiting_for_step = False
        self.next_step_clicked = threading.Event()
        self.step_button = None
        self.step_label = None
        
        # Selected nodes display
        self.selected_nodes_text = None
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Neural Network Visualizer - Live Training")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg='#0a0e17')
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = tk.Frame(self.root, bg='#0a0e17')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_frame, bg='#0a0e17')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            title_frame,
            text="Neural Network Visualizer",
            font=('Helvetica', 20, 'bold'),
            bg='#0a0e17',
            fg='#00ff88'
        )
        title_label.pack(side=tk.LEFT)
        
        # Status frame
        self.status_frame = tk.Frame(title_frame, bg='#0a0e17')
        self.status_frame.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Status: Ready",
            font=('Helvetica', 10),
            bg='#0a0e17',
            fg='#f1f2f6'
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)
        
        # Main content area with canvas and side panel
        content_frame = tk.Frame(main_frame, bg='#0a0e17')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for network visualization
        canvas_frame = tk.Frame(content_frame, bg='#161b22', bd=2, relief=tk.SUNKEN)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = NetworkCanvas(canvas_frame, width=self.width - 340, height=self.height - 200)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Side panel for selected nodes
        side_panel = tk.Frame(content_frame, bg='#161b22', bd=2, relief=tk.SUNKEN, width=280)
        side_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        side_panel.pack_propagate(False)
        
        # Selected nodes header
        tk.Label(
            side_panel,
            text="Selected Nodes",
            font=('Helvetica', 12, 'bold'),
            bg='#161b22',
            fg='#00ff88'
        ).pack(pady=(10, 5))
        
        # Instructions
        tk.Label(
            side_panel,
            text="Click nodes to select\nCtrl+Click for multiple",
            font=('Helvetica', 9),
            bg='#161b22',
            fg='#747d8c',
            justify='center'
        ).pack(pady=(0, 10))
        
        # Scrollable text area for node info
        text_frame = tk.Frame(side_panel, bg='#161b22')
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.selected_nodes_text = tk.Text(
            text_frame,
            bg='#0a0e17',
            fg='#f1f2f6',
            font=('Consolas', 9),
            height=10,
            yscrollcommand=scrollbar.set,
            wrap=tk.WORD
        )
        self.selected_nodes_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.selected_nodes_text.yview)
        
        # Set callback for node selection
        self.canvas.on_node_selected = self._update_selected_nodes_display
        
        # Info panel at bottom
        info_frame = tk.Frame(main_frame, bg='#0a0e17', height=100)
        info_frame.pack(fill=tk.X, pady=(10, 0))
        info_frame.pack_propagate(False)
        
        # Stats boxes
        self.stat_labels = {}
        stats = ['Layers', 'Nodes', 'Edges', 'FPS', 'Updates']
        
        for i, stat in enumerate(stats):
            box = tk.Frame(info_frame, bg='#161b22', bd=1, relief=tk.RAISED)
            box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
            
            label = tk.Label(
                box,
                text=stat,
                font=('Helvetica', 10),
                bg='#161b22',
                fg='#747d8c'
            )
            label.pack(pady=(5, 0))
            
            value = tk.Label(
                box,
                text='-',
                font=('Helvetica', 16, 'bold'),
                bg='#161b22',
                fg='#00ff88'
            )
            value.pack(pady=(0, 5))
            
            self.stat_labels[stat] = value
        
        # Control buttons
        control_frame = tk.Frame(info_frame, bg='#0a0e17')
        control_frame.pack(side=tk.RIGHT, padx=10)
        
        # Step counter label
        self.step_label = tk.Label(
            control_frame,
            text="Step: 0",
            font=('Helvetica', 12, 'bold'),
            bg='#0a0e17',
            fg='#00ff88',
            padx=20
        )
        self.step_label.pack(side=tk.LEFT, padx=5)
        
        # Steps input field
        tk.Label(
            control_frame,
            text="Steps:",
            font=('Helvetica', 10),
            bg='#0a0e17',
            fg='#f1f2f6'
        ).pack(side=tk.LEFT, padx=(10, 5))
        
        self.steps_var = tk.StringVar(value="1")
        self.steps_var.trace_add('write', self._update_step_button_text)
        self.steps_entry = tk.Entry(
            control_frame,
            textvariable=self.steps_var,
            font=('Helvetica', 10),
            bg='#161b22',
            fg='#f1f2f6',
            insertbackground='#f1f2f6',
            width=5,
            justify='center'
        )
        self.steps_entry.pack(side=tk.LEFT, padx=5)
        
        # Next Step button
        self.step_button = tk.Button(
            control_frame,
            text="Next Step",
            command=self._on_next_step,
            bg='#3742fa',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=5,
            state=tk.DISABLED
        )
        self.step_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = tk.Button(
            control_frame,
            text="Pause",
            command=self._toggle_pause,
            bg='#3742fa',
            fg='white',
            font=('Helvetica', 10, 'bold'),
            relief=tk.FLAT,
            padx=20,
            pady=5
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="Reset View",
            command=self._reset_view,
            bg='#2f3542',
            fg='white',
            font=('Helvetica', 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="Reset Zoom",
            command=self._reset_zoom,
            bg='#2f3542',
            fg='white',
            font=('Helvetica', 10),
            relief=tk.FLAT,
            padx=20,
            pady=5
        ).pack(side=tk.LEFT, padx=5)
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
    
    def visualize(self, model: Any, framework: str = 'auto', sample_input: Optional[np.ndarray] = None):
        """Start visualizing a neural network model"""
        self.model = model
        
        # Detect framework
        if framework == 'auto':
            framework = self._detect_framework(model)
        
        # Create parser
        self.parser = self._create_parser(framework)
        
        # Parse network
        self._parse_network(model)
        
        # Draw network
        self.canvas.draw_network(self.layers, self.edges)
        
        # Update stats
        self._update_stats()
        
        # Start update thread
        self.is_running = True
        self.start_time = time.time()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        # Start periodic UI updates
        self._schedule_ui_update()
        
        print(f"Tkinter Neural Network Visualizer started")
        print(f"Network: {len(self.layers)} layers, {len(self.node_map)} nodes, {len(self.edges)} edges")
        
        return self
    
    def _detect_framework(self, model: Any) -> str:
        """Detect model framework"""
        model_class = model.__class__.__module__
        
        if 'torch' in model_class:
            return 'pytorch'
        elif 'tensorflow' in model_class or 'keras' in model_class:
            return 'tensorflow'
        elif hasattr(model, 'named_parameters'):
            return 'pytorch'
        elif hasattr(model, 'layers'):
            return 'tensorflow'
        
        raise ValueError("Could not detect model framework")
    
    def _create_parser(self, framework: str):
        """Create appropriate parser"""
        if framework == 'pytorch':
            return PyTorchParser()
        elif framework == 'tensorflow':
            return TensorFlowParser()
        else:
            raise ValueError(f"Unsupported framework: {framework}")
    
    def _parse_network(self, model: Any):
        """Parse network structure"""
        self.layers = []
        self.edges = []
        self.node_map = {}
        
        layer_index = 0
        prev_layer_nodes = []
        
        # Get modules/layers based on framework
        if hasattr(model, 'named_modules'):
            modules = list(model.named_modules())[1:]
        elif hasattr(model, 'layers'):
            modules = [(layer.name, layer) for layer in model.layers]
        else:
            return
        
        # Filter to only layers with neurons
        valid_modules = []
        for name, module in modules:
            size = self._get_layer_size(module)
            if size > 0:
                valid_modules.append((name, module, size))
        
        num_layers = len(valid_modules)
        
        # Calculate positions
        margin_x = 100
        available_width = self.width - 2 * margin_x
        layer_spacing = available_width / max(num_layers - 1, 1)
        
        for name, module, size in valid_modules:
            layer_id = f"layer_{layer_index}"
            layer_type = module.__class__.__name__
            
            # Limit nodes for visualization
            display_size = min(size, self.max_nodes_per_layer)
            
            # Calculate positions
            layer_x = margin_x + layer_index * layer_spacing
            
            # Calculate vertical spacing
            margin_y = 120
            available_height = self.height - margin_y - 100
            node_spacing = available_height / max(display_size, 1)
            
            nodes = []
            for i in range(display_size):
                node_id = f"{layer_id}_node_{i}"
                y = margin_y + (i + 0.5) * node_spacing
                
                node = Node(
                    id=node_id,
                    layer_id=layer_id,
                    layer_name=name,
                    index=i,
                    x=layer_x,
                    y=y,
                    radius=12
                )
                nodes.append(node)
                self.node_map[node_id] = node
            
            # Create edges from previous layer with actual weights
            if prev_layer_nodes:
                # Try to get weight matrix from the module
                weight_matrix = None
                if hasattr(module, 'weight') and module.weight is not None:
                    weight_matrix = module.weight.detach().cpu().numpy()
                
                for i, prev_node in enumerate(prev_layer_nodes):
                    for j, curr_node in enumerate(nodes):
                        # Get actual weight if available, otherwise use small random
                        if weight_matrix is not None and i < weight_matrix.shape[1] and j < weight_matrix.shape[0]:
                            weight = float(weight_matrix[j, i])
                        else:
                            weight = np.random.randn() * 0.1
                        
                        edge = Edge(
                            source_id=prev_node.id,
                            target_id=curr_node.id,
                            weight=weight,
                            layer_index=layer_index,
                            source_idx=i,
                            target_idx=j
                        )
                        self.edges.append(edge)
            
            layer = Layer(
                id=layer_id,
                name=name,
                layer_type=layer_type,
                size=size,
                index=layer_index,
                x_position=layer_x,
                nodes=nodes,
                params=self._get_layer_params(module)
            )
            self.layers.append(layer)
            
            prev_layer_nodes = nodes
            layer_index += 1
    
    def _get_layer_size(self, module: Any) -> int:
        """Get layer output size"""
        if hasattr(module, 'out_features'):
            return module.out_features
        elif hasattr(module, 'units'):
            return module.units
        elif hasattr(module, 'out_channels'):
            return module.out_channels
        elif hasattr(module, 'weight') and len(module.weight.shape) >= 2:
            return module.weight.shape[0]
        return 0
    
    def _get_layer_params(self, module: Any) -> Dict[str, Any]:
        """Extract layer parameters"""
        params = {}
        if hasattr(module, 'weight') and module.weight is not None:
            params['weight_shape'] = list(module.weight.shape)
        if hasattr(module, 'bias') and module.bias is not None:
            params['has_bias'] = True
        return params
    
    def _update_loop(self):
        """Background thread for processing updates"""
        while self.is_running:
            try:
                update_data = self.update_queue.get(timeout=0.01)
                if update_data is not None:
                    self._apply_update(update_data)
            except queue.Empty:
                pass
            time.sleep(0.001)
    
    def _apply_update(self, update_data: Dict):
        """Apply update to visualization"""
        if 'activations' in update_data:
            self.canvas.update_activations(update_data['activations'])
        
        if 'weights' in update_data:
            self.canvas.update_weights(update_data['weights'])
        
        self.total_updates += 1
    
    def update(self, input_data: Optional[np.ndarray] = None):
        """Update visualization with current model state"""
        if not self.model or not self.parser:
            return
        
        update_data = {}
        
        # Extract weights
        try:
            update_data['weights'] = self.parser.extract_weights(self.model)
        except Exception as e:
            print(f"Error extracting weights: {e}")
        
        if input_data is not None:
            try:
                update_data['activations'] = self.parser.extract_activations(self.model, input_data)
            except Exception as e:
                print(f"Error extracting activations: {e}")
        
        self.update_queue.put(update_data)
    
    def _schedule_ui_update(self):
        """Schedule periodic UI updates"""
        if self.is_running:
            self._update_stats()
            self.root.after(500, self._schedule_ui_update)
    
    def _update_stats(self):
        """Update statistics display"""
        self.stat_labels['Layers'].config(text=str(len(self.layers)))
        self.stat_labels['Nodes'].config(text=str(len(self.node_map)))
        self.stat_labels['Edges'].config(text=str(len(self.edges)))
        self.stat_labels['FPS'].config(text=str(self.canvas.get_fps()))
        self.stat_labels['Updates'].config(text=str(self.total_updates))
        
        # Update selected nodes display with current values
        if self.canvas.selected_nodes:
            self._update_selected_nodes_display(list(self.canvas.selected_nodes.values()))
    
    def _toggle_pause(self):
        """Toggle pause/resume"""
        # Implementation for pausing updates
        pass
    
    def _reset_view(self):
        """Reset view"""
        # Redraw network
        self.canvas.draw_network(self.layers, self.edges)
    
    def _reset_zoom(self):
        """Reset zoom to see entire network"""
        self.canvas.reset_zoom()
    
    def _update_selected_nodes_display(self, selected_nodes: List[Node]):
        """Update the side panel with selected node information"""
        if not self.selected_nodes_text:
            return
        
        self.selected_nodes_text.delete('1.0', tk.END)
        
        if not selected_nodes:
            self.selected_nodes_text.insert(tk.END, "No nodes selected\n")
            self.selected_nodes_text.insert(tk.END, "Click on nodes to view their outputs")
            return
        
        self.selected_nodes_text.insert(tk.END, f"Selected: {len(selected_nodes)} node(s)\n")
        self.selected_nodes_text.insert(tk.END, "=" * 30 + "\n\n")
        
        for node in selected_nodes:
            self.selected_nodes_text.insert(tk.END, f"Node: {node.id}\n")
            self.selected_nodes_text.insert(tk.END, f"Layer: {node.layer_name}\n")
            self.selected_nodes_text.insert(tk.END, f"Index: {node.index}\n")
            self.selected_nodes_text.insert(tk.END, f"Output: {node.activation:.6f}\n")
            if node.bias != 0:
                self.selected_nodes_text.insert(tk.END, f"Bias: {node.bias:.6f}\n")
            self.selected_nodes_text.insert(tk.END, "-" * 30 + "\n\n")
    
    def enable_step_mode(self, enabled: bool = True):
        """Enable or disable step-by-step mode"""
        self.step_mode = enabled
        if enabled:
            self.step_button.config(state=tk.NORMAL)
            print("Step-by-step mode enabled. Click 'Next Step' to advance.")
        else:
            self.step_button.config(state=tk.DISABLED)
            self.next_step_clicked.set()  # Release any waiting threads
            print("Automatic mode enabled.")
    
    def _on_next_step(self):
        """Handle Next Step button click"""
        self.next_step_clicked.set()
        self.step_button.config(text="Processing...", state=tk.DISABLED)
    
    def get_step_count(self) -> int:
        """Get the number of steps to run from the input field"""
        try:
            count = int(self.steps_var.get())
            return max(1, count)  # Minimum 1 step
        except ValueError:
            return 1
    
    def _update_step_button_text(self, *args):
        """Update button text when step count changes"""
        if not self.step_mode or not self.step_button:
            return
        
        steps = self.get_step_count()
        if steps > 1:
            self.step_button.config(text=f"Next ({steps})")
        else:
            self.step_button.config(text="Next Step")
    
    def wait_for_next_step(self) -> int:
        """Block until Next Step button is clicked, returns number of steps to run"""
        if not self.step_mode:
            return 1
        
        self.waiting_for_step = True
        self.next_step_clicked.clear()
        
        # Update UI to show waiting state
        self.step_button.config(text="Next Step", state=tk.NORMAL)
        self.status_label.config(text="Status: Waiting for next step(s)...")
        
        # Wait for button click (non-blocking of main thread)
        while not self.next_step_clicked.is_set() and self.is_running:
            time.sleep(0.01)
        
        # Get the requested number of steps AFTER button was clicked
        steps_to_run = self.get_step_count()
        
        # Immediately clear the event so we don't trigger again
        self.next_step_clicked.clear()
        
        self.waiting_for_step = False
        return steps_to_run
    
    def run(self):
        """Start the main loop"""
        self.root.mainloop()
    
    def stop(self):
        """Stop the visualizer"""
        self.is_running = False
        self.next_step_clicked.set()  # Release any waiting threads
        if self.update_thread:
            self.update_thread.join(timeout=1.0)
        self.root.destroy()


# Example and demo
if __name__ == '__main__':
    print("Tkinter Neural Network Visualizer")
    print("=================================")
    print()
    
    try:
        import torch
        import torch.nn as nn
        
        # Create model
        model = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )
        
        print("Creating visualizer...")
        viz = TkinterNeuralVisualizer(width=1400, height=800)
        viz.visualize(model, framework='pytorch')
        
        # Enable step-by-step mode
        viz.enable_step_mode(True)
        
        # Training simulation in separate thread
        def train():
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            step = 0
            while step < 10000 and viz.is_running:
                # Wait for user to click Next Step and get number of steps to run
                steps_to_run = viz.wait_for_next_step()
                
                # Run multiple steps if requested
                for i in range(steps_to_run):
                    if not viz.is_running or step >= 10000:
                        break
                    
                    inputs = torch.randn(32, 20)
                    targets = torch.randn(32, 10)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    # Update visualization
                    viz.update(inputs.numpy())
                    
                    step += 1
                    viz.step_count = step
                    
                    # Schedule UI update on main thread
                    viz.root.after(0, lambda s=step: viz.step_label.config(text=f"Step: {s}"))
                    
                    print(f"Step {step} complete ({i+1}/{steps_to_run}). Loss: {loss.item():.4f}")
                    
                    # Small delay to allow UI to update between steps
                    time.sleep(0.05)
        
        train_thread = threading.Thread(target=train, daemon=True)
        train_thread.start()
        
        print("Step-by-step mode enabled.")
        print("Click 'Next Step' button to advance each training step.")
        print("(Close window to stop)")
        viz.run()
        
    except ImportError:
        print("PyTorch not installed. Install with: pip install torch")
        print("Creating demo with random data...")
        
        # Create a mock visualizer without actual model
        viz = TkinterNeuralVisualizer(width=1200, height=700)
        
        # Mock data
        viz.layers = [
            Layer("l0", "Input", "Linear", 784, 0, 100, [Node(f"n{i}", "l0", "Input", i, 100, 100 + i*20) for i in range(10)]),
            Layer("l1", "Hidden1", "Linear", 128, 1, 400, [Node(f"n{i}", "l1", "Hidden1", i, 400, 100 + i*50) for i in range(8)]),
            Layer("l2", "Output", "Linear", 10, 2, 700, [Node(f"n{i}", "l2", "Output", i, 700, 200 + i*60) for i in range(6)]),
        ]
        
        viz.canvas.draw_network(viz.layers, [])
        viz.run()
