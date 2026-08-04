from pre_process.pre_process import preprocess_data
from model.deepmodrp import run_model
from statistics import mean

def main():
    # preprocessing
    print("Step 1: Preprocessing datasets")
    preprocess_data()
    print("Training, testing and evaluation datasets created")
    print("Step 1 complete ✓")

    # training, validation and evaluation
    print("Step 2: Training and evaluating model")
    
    #hyperparameters
    epochs = 300
    batch_size = 1024
    learning_rate = 1e-4
    patience = 30
    
    rmse,pcc, r2, mape = run_model(epochs, batch_size, learning_rate, patience) 
    print("Step 2 complete ✓")

    # results
    print("Results from testing, validation and evaluating model")
    print(f'RMSE: {mean(rmse)}')
    print(f'PCC: {mean(pcc)}')
    print(f'R^2: {mean(r2)}')
    print(f'MAPE: {mean(mape)}')

if __name__ == "__main__":
    main()