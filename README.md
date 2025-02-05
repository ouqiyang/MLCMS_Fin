<<<<<<< HEAD
For building the environment, first run: 
=======
First, change to the project root directory.
For building the environment, run: 
>>>>>>> db1835fb1d8de97a2c4a586f76cc0bd7336ee7fa

    pip install -r code/requirements.txt

For training, testing and evaluating the model,  use:

    python code/main.py
 
 Edit code/ src/ train.py to change the input, replace the following "data/Corridor_Data" with your data path.

    train_loader, test_loader = get_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)
    train_loader_wiedmann, _ = get_wiedmann_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)
<<<<<<< HEAD

For ploting the data, use: 

    python code/src/plot_predictions_ann.py 
    python code/src/plot_predictions.py
=======
>>>>>>> db1835fb1d8de97a2c4a586f76cc0bd7336ee7fa
