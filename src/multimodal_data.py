import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MultimodalEmbeddingDataset(Dataset):
    def __init__(self, x, y, class_id=None, class_embeds=None):
        self.x = x
        self.y = y
        self.n = len(self.x)
        self.zero_shot = not (class_id is None or class_embeds is None)

        if self.zero_shot:
            self.z = class_id
            self.class_embeds = class_embeds

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.zero_shot:
            return i, self.x[i], self.y[i], self.z[i]
        else:
            return i, self.x[i], self.y[i]
    
def load_glove_embeddings(root, img_embed):
    image_features = np.load(os.path.join(root, f"{img_embed}_image_features.npy"))

    train_idx = np.load(os.path.join(root, f"glove_train_idx.npy"))
    val_idx = np.load(os.path.join(root, f"glove_val_idx.npy"))

    x_train = image_features[train_idx]
    x_test = image_features[val_idx]
    y_train = np.load(os.path.join(root, f"glove_train_embeds.npy"))
    y_test = np.load(os.path.join(root, f"glove_val_embeds.npy"))

    val_class_id = np.load(os.path.join(root, f"glove_val_class_id_labels.npy"))
    val_class_embeds = np.load(os.path.join(root, f"glove_val_class_embeds.npy"))

    return x_train, x_test, y_train, y_test, val_class_id, val_class_embeds

def get_multimodal_dataloaders(
    batch_size, 
    rank,
    img_embed,
    txt_embed,
    root="<path/to/imagenet>", 
):
    image_features = np.load(os.path.join(root, f"{img_embed}_image_features.npy"))
    text_features  = np.load(os.path.join(root, f"{txt_embed}_text_features.npy"))
    x_train, x_test, y_train, y_test = train_test_split(image_features, text_features, test_size=0.1, random_state=42)
    test_dataset = MultimodalEmbeddingDataset(x_test, y_test)

    train_dataset = MultimodalEmbeddingDataset(x_train, y_train)
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(train_dataset):>5,} training samples on rank {rank}.")
    test_dataloader = DataLoader(
        test_dataset, shuffle=True, batch_size=batch_size
    )
    print(f"{len(test_dataset):>5,} validation samples on rank {rank}.")
    return train_dataloader, test_dataloader