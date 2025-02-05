import os

if __name__ == "__main__":
    print("1. training model...")
    os.system("python src/train.py")
    
    print("2. testing model...")
    os.system("python src/test.py")

    print("3. Generating training loss plot...")
    os.system("python src/plot_loss.py")
    
    print("4. Generating prediction comparison plot...")
    os.system("python src/plot_predictions.py")
