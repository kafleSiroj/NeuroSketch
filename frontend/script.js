let layers = [];
let selectedLayer = null;
let canvas, ctx;
let canvasWidth, canvasHeight;
const layerSpacing = 120; // Horizontal spacing between layers
const verticalPadding = 60;
const nodeRadius = 8;

const workspace = document.getElementById('workspace');
const canvasHint = document.getElementById('canvasHint');
const layerInfoPanel = document.getElementById('layerInfoPanel');
const layerInfoTitle = document.getElementById('layerInfoTitle');
const layerInfoContent = document.getElementById('layerInfoContent');

// Layer configurations
const layerConfigs = {
    input: {
        name: 'Input',
        color: '#00d4ff',
        defaultNeurons: 784
    },
    dense: {
        name: 'Dense',
        color: '#667eea',
        defaultNeurons: 128
    },
    relu: {
        name: 'ReLU',
        color: '#ff8844',
        defaultNeurons: null
    },
    conv: {
        name: 'Conv2D',
        color: '#44ff88',
        defaultNeurons: 32
    },
    pool: {
        name: 'MaxPool',
        color: '#ff44ff',
        defaultNeurons: null
    },
    output: {
        name: 'Output',
        color: '#ffdd44',
        defaultNeurons: 10
    }
};

// Initialize canvas
function initCanvas() {
    canvas = document.getElementById('networkCanvas');
    ctx = canvas.getContext('2d');
    
    function resizeCanvas() {
        canvasWidth = workspace.clientWidth;
        canvasHeight = workspace.clientHeight;
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        redraw();
    }
    
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    
    // Handle canvas click to select layer or add neuron
    canvas.addEventListener('click', (e) => {
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        for (let i = 0; i < layers.length; i++) {
            const positions = getNeuronPositions(i);
            
            // Check if clicked on a neuron
            for (let pos of positions) {
                const dist = Math.hypot(x - pos.x, y - pos.y);
                if (dist < nodeRadius + 5) {
                    selectedLayer = i;
                    showLayerInfo(i);
                    redraw();
                    return;
                }
            }
        }
        
        // If no neuron clicked, deselect
        selectedLayer = null;
        layerInfoPanel.style.display = 'none';
        redraw();
    });
}

// Add layer
function addLayer(type) {
    const config = layerConfigs[type];
    const layer = {
        id: `layer-${layers.length}`,
        type: type,
        name: config.name,
        neurons: config.defaultNeurons,
        color: config.color,
        x: layers.length * layerSpacing + 50,
    };
    layers.push(layer);
    redraw();
    updateHint();
}

// Calculate neuron positions
function getNeuronPositions(layerIndex) {
    const layer = layers[layerIndex];
    if (!layer.neurons) return [];
    
    const positions = [];
    const neuronCount = Math.min(layer.neurons, 20); // Max 20 neurons visible
    const startY = (canvasHeight - (neuronCount - 1) * 30) / 2;
    
    for (let i = 0; i < neuronCount; i++) {
        positions.push({
            x: layer.x,
            y: startY + i * 30,
            index: i,
            totalNeurons: layer.neurons
        });
    }
    return positions;
}

// Draw network
function redraw() {
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
    
    if (layers.length === 0) return;
    
    // Draw connections
    drawConnections();
    
    // Draw neurons
    for (let i = 0; i < layers.length; i++) {
        drawLayer(i);
    }
}

// Draw connections between layers
function drawConnections() {
    for (let i = 0; i < layers.length - 1; i++) {
        const currentPositions = getNeuronPositions(i);
        const nextPositions = getNeuronPositions(i + 1);
        
        ctx.strokeStyle = 'rgba(100, 150, 255, 0.2)';
        ctx.lineWidth = 1;
        
        currentPositions.forEach(curr => {
            nextPositions.forEach(next => {
                ctx.beginPath();
                ctx.moveTo(curr.x + 25, curr.y);
                ctx.lineTo(next.x - 25, next.y);
                ctx.stroke();
            });
        });
    }
}

// Draw layer
function drawLayer(layerIndex) {
    const layer = layers[layerIndex];
    const positions = getNeuronPositions(layerIndex);
    const isSelected = selectedLayer === layerIndex;
    
    // Draw layer label
    ctx.fillStyle = layer.color;
    ctx.font = 'bold 12px Arial';
    ctx.fillText(layer.name, layer.x - 20, 30);
    
    if (layer.neurons) {
        ctx.font = '10px Arial';
        ctx.fillStyle = '#999';
        ctx.fillText(`${layer.neurons}N`, layer.x - 15, 45);
    }
    
    // Draw neurons
    positions.forEach(pos => {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, nodeRadius, 0, Math.PI * 2);
        ctx.fillStyle = layer.color;
        ctx.fill();
        
        if (isSelected) {
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
    });
    
    // Draw layer box
    if (isSelected) {
        ctx.strokeStyle = layer.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        const minY = positions.length > 0 ? Math.min(...positions.map(p => p.y)) : 0;
        const maxY = positions.length > 0 ? Math.max(...positions.map(p => p.y)) : 0;
        ctx.strokeRect(layer.x - 30, minY - 20, 60, maxY - minY + 40);
        ctx.setLineDash([]);
    }
}

// Show layer info
function showLayerInfo(layerIndex) {
    const layer = layers[layerIndex];
    layerInfoTitle.textContent = layer.name;
    
    let content = `<strong>Type:</strong> ${layer.type}<br>`;
    if (layer.neurons) {
        content += `<strong>Neurons:</strong> ${layer.neurons}<br>`;
        content += `<button onclick="addNeuron(${layerIndex})" style="width:100%;padding:4px;margin-top:5px;background:#667eea;border:none;color:white;cursor:pointer;border-radius:4px;font-size:11px;">+ Add Neuron</button>`;
        content += `<button onclick="removeNeuron(${layerIndex})" style="width:100%;padding:4px;margin-top:3px;background:#ff6b6b;border:none;color:white;cursor:pointer;border-radius:4px;font-size:11px;">− Remove Neuron</button>`;
    }
    content += `<button onclick="deleteLayer(${layerIndex})" style="width:100%;padding:4px;margin-top:3px;background:#ff4444;border:none;color:white;cursor:pointer;border-radius:4px;font-size:11px;">🗑️ Delete Layer</button>`;
    
    layerInfoContent.innerHTML = content;
    layerInfoPanel.style.display = 'block';
}

// Add neuron
function addNeuron(layerIndex) {
    if (layers[layerIndex].neurons) {
        layers[layerIndex].neurons++;
        showLayerInfo(layerIndex);
        redraw();
    }
}

// Remove neuron
function removeNeuron(layerIndex) {
    if (layers[layerIndex].neurons && layers[layerIndex].neurons > 1) {
        layers[layerIndex].neurons--;
        showLayerInfo(layerIndex);
        redraw();
    }
}

// Delete layer
function deleteLayer(layerIndex) {
    layers.splice(layerIndex, 1);
    // Recalculate layer IDs and positions
    layers = layers.map((layer, idx) => ({
        ...layer,
        id: `layer-${idx}`,
        x: idx * layerSpacing + 50
    }));
    selectedLayer = null;
    layerInfoPanel.style.display = 'none';
    redraw();
    updateHint();
}

// Update hint visibility
function updateHint() {
    if (layers.length > 0) {
        workspace.classList.add('has-network');
        canvasHint.style.display = 'none';
    } else {
        workspace.classList.remove('has-network');
        canvasHint.style.display = 'flex';
    }
}

// Clear workspace
function clearWorkspace() {
    if (layers.length > 0 && confirm('Clear all layers?')) {
        layers = [];
        selectedLayer = null;
        layerInfoPanel.style.display = 'none';
        redraw();
        updateHint();
    }
}

// Train network
function trainNetwork() {
    if (layers.length === 0) {
        alert('Please add at least one layer!');
        return;
    }
    const summary = layers.map((l, i) => `${i + 1}. ${l.name}${l.neurons ? ` (${l.neurons}N)` : ''}`).join('\n');
    alert(`🚀 Training Network\n\n${summary}\n\nStarting training...`);
}

// Attach layer template listeners
document.querySelectorAll('.layer-template').forEach(template => {
    template.addEventListener('click', () => {
        const type = template.dataset.type;
        addLayer(type);
    });
});

// Initialize
initCanvas();
updateHint();
