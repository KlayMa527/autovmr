import torch

# The refactored function
def process_image_token_index(labels, image_token_index, replacement_length=100):
    labels = labels.clone()
    indices = (labels == image_token_index).nonzero(as_tuple=True)
    
    if not indices[0].size(0):
        return labels

    new_labels_list = []
    for row, col in zip(*indices):
        new_label = torch.cat([
            labels[row, :col],
            torch.full((replacement_length,), image_token_index, device=labels.device),
            labels[row, col + 1:]
        ])
        new_labels_list.append(new_label.unsqueeze(0))

    return torch.cat(new_labels_list, dim=0)

# Define test labels tensor
labels = torch.tensor([[1, 2, 3, -200, 5, 6],
                       [1, 2, 4, -200, 5, 6],
                       [1, 2, 3, -200, 6, 7]])

# Define the token to be replaced and replacement length
image_token_index = -200
replacement_length = 100  # Smaller length for easier visualization

# Process the labels
processed_labels = process_image_token_index(labels, image_token_index, replacement_length)

# Show the original and processed labels
print("Original labels:")
print(labels)
print("Processed labels:")
print(processed_labels)
