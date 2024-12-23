from pathlib import Path
from lightning import Trainer
import torch

from dataset import FieldDataModule
from model import SequenceClassificationTask
from cfg import root, batch_size, hidden_size, num_layers, learning_rate, max_epochs, min_epochs, early_stopping_patience, gpu, experiment_name


dm = FieldDataModule(root=root, batch_size=batch_size, workers=2)


# Create the task with the sampled hyperparameters
task = SequenceClassificationTask(input_size=512,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  learning_rate=learning_rate)

# Create a dedicated models' directory for saving the trial's best models
models_path = Path(f"./models/{experiment_name}/")
models_path.mkdir(parents=True, exist_ok=True)

# Set the callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=models_path,
    filename=f"model-{{epoch:02d}}-{{val_loss:.2f}}",
    save_top_k=1,
    mode="min",
)
early_stopping_callback = EarlyStopping(monitor="val_loss",
                                        mode="min",
                                        patience=early_stopping_patience)

# Create a TensorBoard logger
logger = TensorBoardLogger("./tb_logs", name=experiment_name)

# Trainer definition
trainer = Trainer(
    logger=logger,
    accelerator='gpu',
    devices=[gpu],
    max_epochs=max_epochs,
    min_epochs=min_epochs,
    callbacks=[checkpoint_callback, early_stopping_callback],
    precision=16
)

trainer.fit(model=task, datamodule=dm)

checkpoint_callback.best_model_score.item()


#  Load your model
model = SequenceClassificationTask.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                                        input_size=512,
                                                        hidden_size=hidden_size)
model.eval()
model.freeze()

# Get the validation data loader
test_dl = dm.test_dataloader()

# Predict
all_logits = []
y_tests = []

with torch.no_grad():
    for batch in test_dl:
        inputs = batch['image']
        y_test = batch['label']
        logits = model(inputs)
        all_logits.append(logits)
        y_tests.append(y_test)

# Concatenate all the results
all_logits = torch.cat(all_logits, dim=0)
y_test = torch.cat(y_tests, dim=0)

# Get the probabilities
y_test_hat = torch.nn.functional.softmax(all_logits, dim=1)


from sklearn.calibration import label_binarize
from sklearn.metrics import log_loss

# Get the arrays
y_test_np = y_test.cpu().numpy()
y_test_hat_np = y_test_hat.cpu().numpy()

# Convert y_val to a binary label indicator format
y_test_bin = label_binarize(y_test_np, classes=[0, 1, 2, 3, 4, 5, 6])

cross_entropy = log_loss(y_test_bin, y_test_hat_np)
print("Cross Entropy:", cross_entropy)