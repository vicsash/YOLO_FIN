if __name__ == '__main__':
    import os
    from datetime import datetime
    from ultralytics import YOLO
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    from PIL import Image

    class EarlyStopping:
        def __init__(self, patience=3, verbose=False):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_loss = None
            self.early_stop = False

        def __call__(self, val_loss):
            if self.best_loss is None:
                self.best_loss = val_loss
            elif val_loss > self.best_loss:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_loss = val_loss
                self.counter = 0

    def polygon_to_bbox(polygon):
        polygon = np.array(polygon)
        x_min = np.min(polygon[:, 0])
        y_min = np.min(polygon[:, 1])
        x_max = np.max(polygon[:, 0])
        y_max = np.max(polygon[:, 1])

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        return x_center, y_center, width, height

    def preprocess_annotations(annotation_file):
        bboxes = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                if len(parts) == 5:
                    x_center, y_center, width, height = map(float, parts[1:])
                    bboxes.append((class_id, x_center, y_center, width, height))
                else:
                    polygon = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts), 2)]
                    x_center, y_center, width, height = polygon_to_bbox(polygon)
                    bboxes.append((class_id, x_center, y_center, width, height))
        return bboxes

    def resize_image_and_bboxes(image, bboxes, target_size):
        original_size = image.size
        image = image.resize(target_size, Image.ANTIALIAS)

        scale_x = target_size[0] / original_size[0]
        scale_y = target_size[1] / original_size[1]

        resized_bboxes = []
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            x_center *= scale_x
            y_center *= scale_y
            width *= scale_x
            height *= scale_y
            resized_bboxes.append((class_id, x_center, y_center, width, height))

        return image, resized_bboxes

    class CustomDataset(Dataset):
        def __init__(self, image_dir, annotation_dir, transform=None, target_size=(512, 512), device='cpu'):
            self.image_dir = image_dir
            self.annotation_dir = annotation_dir
            self.transform = transform
            self.target_size = target_size
            self.image_files = os.listdir(image_dir)
            self.annotation_files = os.listdir(annotation_dir)
            self.device = device

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image_path = os.path.join(self.image_dir, self.image_files[idx])
            annotation_path = os.path.join(self.annotation_dir, self.annotation_files[idx])
            image = Image.open(image_path).convert('RGB')
            bboxes = preprocess_annotations(annotation_path)
            image, bboxes = resize_image_and_bboxes(image, bboxes, self.target_size)
            if self.transform:
                image = self.transform(image)
            image = image.to(self.device)
            bboxes = [torch.tensor(bbox).to(self.device) for bbox in bboxes]
            return image, bboxes

    # Generate a unique directory name based on the current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    directory = f'runs/train/exp_{timestamp}'

    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory created: {directory}")
    else:
        print(f"Directory already exists: {directory}")

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize and move the model to the GPU
    model = YOLO('yolov8m.pt').to(device)

    # Add dropout to the model
    def add_dropout(model):
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.dropout = torch.nn.Dropout(p=0.5)

    add_dropout(model)

    early_stopping = EarlyStopping(patience=3, verbose=True)
    writer = SummaryWriter(log_dir=directory)

    # Preprocess annotations
    train_annotations = 'Card_Detection.v10i.yolov8/train/labels'
    val_annotations = 'Card_Detection.v10i.yolov8/valid/labels'
    test_annotations = 'Card_Detection.v10i.yolov8/test/labels'

    # Create datasets and dataloaders
    train_dataset = CustomDataset('Card_Detection.v10i.yolov8/train/images', train_annotations, device=device)
    val_dataset = CustomDataset('Card_Detection.v10i.yolov8/valid/images', val_annotations, device=device)
    test_dataset = CustomDataset('Card_Detection.v10i.yolov8/test/images', test_annotations, device=device)

    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

    # Train the model for 10 epochs using your dataset with image size 640
    results = model.train(data='Card_Detection.v10i.yolov8/data.yaml', epochs=100, imgsz=512,)

    # Save the model
    model_save_path = os.path.join(directory, 'model.pt')
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")