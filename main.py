import os

if __name__ == "__main__":
    print("1. training model...")
    os.system("python src/train.py")
    
    print("2. testing model...")
    os.system("python src/test.py")
