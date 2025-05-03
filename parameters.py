# Loading pre-trained YOLO model (Example: trying yolov8x)
model = YOLO("yolov8x.pt") # Try XtraLarge model
print("Loaded pre-trained yolov8x.pt model.")

# Hyperparameters - Example with adjustments
print("Starting model training with adjusted parameters...")
results = model.train(
    data=yaml_path,
    epochs=75,                 # INCREASED epochs
    batch=16,                  # Keep or adjust based on GPU memory
    imgsz=640,
    patience=20,
    optimizer='AdamW',         # CHANGED optimizer (try AdamW)
    # momentum=0.937,          # Not needed for AdamW
    lr0=0.0001,                # REDUCED learning rate (typical for AdamW)
    weight_decay=0.0001,       # Keep or slightly adjust
    cos_lr=True,
    save_period=-1,
    workers=2,
    cache=True,                # ADDED cache for potential speedup
    plots=True,                # ADDED automatic plot generation

    # Augmentations - Example with slight adjustments
    hsv_h=0.015,
    hsv_s=0.5,                 # REDUCED saturation augmentation
    hsv_v=0.3,                 # REDUCED value augmentation
    flipud=0.3,                # Maybe slightly less flipud? Experiment.
    fliplr=0.5,
    translate=0.1,
    scale=0.5,
    shear=0.01,
    # copy_paste=0.1,          # ADDED CopyPaste augmentation (optional)


    # Define project and run name (use a new name for the new experiment)
    project='S2R2_Detection',
    name='yolov8x_run_longer_adamw' # NEW name for this run
)

print("Training finished.")
# Use the analysis code block after this cell runs
