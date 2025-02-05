First, change to the project root directory.
For building the environment, first run: 

    pip install -r code/requirements.txt

For the whole process of training, testing and evaluating the model,  use:

    python code/main.py
 
 Edit code/ src/ train.py to change the input, replace the following "data/Corridor_Data" with your data path.

    train_loader, test_loader = get_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)
    train_loader_wiedmann, _ = get_wiedmann_dataloader("data/Corridor_Data", batch_size=BATCH_SIZE)
