# Neural Network Visualizer

A high-performance, customizable neural network visualization tool with smooth animations and modern styling.

## Features

- **Universal Model Support**: Works with PyTorch and TensorFlow/Keras models
- **Real-time Updates**: Live visualization during training with minimal overhead
- **Interactive Controls**: Pause, reset, zoom, and explore the network
- **Activation Visualization**: Color-coded nodes show positive/negative/zero activations
- **Weight Display**: Visual representation of connection strengths (green=positive, red=negative)
- **Layer Information**: Detailed stats about each layer
- **Performance Metrics**: FPS counter and update statistics
- **Step-by-Step Mode**: Train one step at a time with manual advancement

## Installation

```bash
# Required dependencies
pip install numpy

# For PyTorch support
pip install torch

# For TensorFlow support
pip install tensorflow
```

## Quick Start

```python
import torch
import torch.nn as nn
from tkinter_neural_viz import TkinterNeuralVisualizer

# Define your model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Create and start visualizer
viz = TkinterNeuralVisualizer(width=1400, height=800)
viz.visualize(model, framework='pytorch')

# During training
for epoch in range(100):
    for inputs, targets in dataloader:
        # ... training code ...
        
        # Update visualization (every few steps)
        viz.update(inputs.numpy())

viz.run()  # Start the GUI
```

## Framework Support

### PyTorch

```python
import torch
import torch.nn as nn
from tkinter_neural_viz import TkinterNeuralVisualizer

model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

viz = TkinterNeuralVisualizer()
viz.visualize(model, framework='pytorch')
```

### TensorFlow/Keras

```python
import tensorflow as tf
from tkinter_neural_viz import TkinterNeuralVisualizer

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

viz = TkinterNeuralVisualizer()
viz.visualize(model, framework='tensorflow')
```

## Step-by-Step Mode

Enable manual training progression:

```python
viz.enable_step_mode(True)

for step in range(1000):
    # Wait for user to click "Next Step"
    steps_to_run = viz.wait_for_next_step()
    
    # Run the requested number of steps
    for i in range(steps_to_run):
        # ... training code ...
        viz.update(inputs.numpy())
```

## Interactive Features

- **Click nodes** to see their output values in the side panel
- **Ctrl+Click** to select multiple nodes
- **Drag to zoom** to a specific region
- **Reset Zoom** button to see the entire network
- **Next Step** button for manual training progression

## Configuration

```python
viz = TkinterNeuralVisualizer(
    width=1600,              # Window width
    height=900,              # Window height
    max_nodes_per_layer=32   # Limit nodes for large layers
)
```

## Troubleshooting

### Display Issues

**Problem**: `No display name and no $DISPLAY environment variable`
**Solution**: 
- On remote servers, use X11 forwarding: `ssh -X user@host`
- Or run on a machine with a display/GUI environment

### Performance Issues

**Problem**: Low FPS or choppy animation
**Solutions**:
1. Reduce `max_nodes_per_layer` to show fewer nodes per layer
2. Update less frequently (every 10-20 training steps instead of every step)

## License

MIT License - Feel free to use in your projects!
# neural-viz
