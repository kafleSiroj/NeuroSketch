let draggedElement = null;
let draggedFromSidebar = false;

const sidebar = document.querySelector('.sidebar');
const workspace = document.querySelector('#workspace');

// Sidebar drag and drop
sidebar.addEventListener('dragstart', (e) => {
    if (e.target.classList.contains('block')) {
        draggedElement = e.target;
        draggedFromSidebar = true;
        e.target.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'copy';
        e.dataTransfer.setData('text/html', e.target.innerHTML);
    }
});

sidebar.addEventListener('dragend', (e) => {
    if (draggedElement) {
        draggedElement.classList.remove('dragging');
        draggedElement = null;
        draggedFromSidebar = false;
    }
});

// Workspace drag over and drop
workspace.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    workspace.classList.add('drag-over');
});

workspace.addEventListener('dragleave', (e) => {
    if (e.target === workspace) {
        workspace.classList.remove('drag-over');
    }
});

workspace.addEventListener('drop', (e) => {
    e.preventDefault();
    workspace.classList.remove('drag-over');

    if (draggedFromSidebar && draggedElement) {
        // Create a new block in the workspace
        const newBlock = document.createElement('div');
        newBlock.className = `dropped-block ${draggedElement.className.split(' ').pop()}`;
        newBlock.draggable = true;
        newBlock.textContent = draggedElement.textContent;

        // Add delete button
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'delete-btn';
        deleteBtn.innerHTML = '×';
        deleteBtn.addEventListener('click', () => {
            newBlock.remove();
            if (workspace.children.length === 1) {
                workspace.classList.remove('has-blocks');
            }
        });
        newBlock.appendChild(deleteBtn);

        workspace.appendChild(newBlock);
        workspace.classList.add('has-blocks');

        // Make the new block draggable within workspace
        makeWorkspaceBlockDraggable(newBlock);
    }
});

// Workspace block repositioning
function makeWorkspaceBlockDraggable(block) {
    let offsetX = 0;
    let offsetY = 0;

    block.addEventListener('dragstart', (e) => {
        draggedFromSidebar = false;
        draggedElement = block;
        block.classList.add('dragging-workspace');
        e.dataTransfer.effectAllowed = 'move';

        const rect = block.getBoundingClientRect();
        const workspaceRect = workspace.getBoundingClientRect();
        offsetX = e.clientX - rect.left;
        offsetY = e.clientY - rect.top;
    });

    block.addEventListener('dragend', (e) => {
        block.classList.remove('dragging-workspace');
        draggedElement = null;
    });
}

// Category collapse/expand
const categoryTitles = document.querySelectorAll('.category-title');
categoryTitles.forEach((title) => {
    title.addEventListener('click', () => {
        const container = title.nextElementSibling;
        title.classList.toggle('expanded');
        title.classList.toggle('collapsed');
        container.classList.toggle('hidden');
    });
});

// Workspace repositioning for workspace blocks
workspace.addEventListener('dragover', (e) => {
    if (!draggedFromSidebar && draggedElement && draggedElement.classList.contains('dropped-block')) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';

        // Visual feedback for workspace blocks
        const rect = workspace.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Position follows the cursor
        draggedElement.style.position = 'absolute';
        draggedElement.style.left = (x - 60) + 'px';
        draggedElement.style.top = (y - 20) + 'px';
    }
});

workspace.addEventListener('drop', (e) => {
    if (!draggedFromSidebar && draggedElement && draggedElement.classList.contains('dropped-block')) {
        e.preventDefault();
        draggedElement.style.position = 'static';
    }
});
