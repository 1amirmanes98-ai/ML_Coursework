import numpy as np
import matplotlib.pyplot as plt
from backprop_network import Network
from backprop_data import load_as_matrix_with_labels

# --- HELPER FUNCTIONS ---

def run_experiment_batch(learning_rates, x_train, y_train, x_test, y_test, epochs, batch_size, dims):
    """
    Runs training for a list of learning rates and returns a results dictionary.
    """
    results = {}
    for lr in learning_rates:
        print(f"\nTraining with Learning Rate: {lr}")
        
        # Initialize a FRESH network for every run
        net = Network(dims)
        
        # Train
        _, train_loss, _, train_acc, test_acc = net.train(
            x_train, y_train, epochs, batch_size, lr, x_test, y_test
        )
        
        # Store results
        results[lr] = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        }
    return results

def plot_metrics(results, epochs, metric_key, title, ylabel, y_limit=None):
    """
    Generic plotting function for multiple lines (Task B style).
    """
    plt.figure(figsize=(10, 6))
    epochs_range = range(1, epochs + 1)
    
    for lr, data in results.items():
        plt.plot(epochs_range, data[metric_key], label=f'LR={lr}')
        
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel(ylabel)
    if y_limit:
        plt.ylim(y_limit)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_accuracy_comparison(train_acc, test_acc, title, vlines=None):
    """
    Plots Training vs Test Accuracy on a single graph.
    Optional: vlines list of (x_position, label) for marking events (like LR drops).
    """
    plt.figure(figsize=(10, 6))
    epochs = len(train_acc)
    
    plt.plot(range(1, epochs + 1), train_acc, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), test_acc, label="Test Accuracy", linewidth=2)
    
    if vlines:
        for x_pos, label in vlines:
            plt.axvline(x=x_pos, color='r', linestyle=':', label=label)
            
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_weights(W, title):
    """
    Plots the weight matrices as images (for Task D).
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle(title)
    
    for i, ax in enumerate(axes.flat):
        # Reshape the i-th row into 28x28 image
        img = W[i, :].reshape(28, 28)
        
        # Plot using 'nearest' interpolation
        ax.imshow(img, interpolation='nearest', cmap='viridis')
        ax.set_title(f"Digit {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# --- MAIN TASKS ---

########################## Section B ##############################

def task_b_learning_rates():
    print("\n========== TASK B: Learning Rate Experiments ==========")
    # 1. Load Data (Subset 10k)
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(10000, 5000)

    # 2. Configuration
    learning_rates = [0.001, 0.01, 0.1, 1, 10]
    epochs = 30
    batch_size = 10
    dims = [784, 40, 10]

    # 3. Run Experiments
    results = run_experiment_batch(learning_rates, x_train, y_train, x_test, y_test, epochs, batch_size, dims)

    # 4. Generate Plots
    plot_metrics(results, epochs, 'train_acc', "Task B: Training Accuracy", "Accuracy")
    plot_metrics(results, epochs, 'train_loss', "Task B: Training Loss", "Loss", y_limit=(0, 2.5))
    plot_metrics(results, epochs, 'test_acc', "Task B: Test Accuracy", "Accuracy")

########################## Section C ##############################

def task_c_full_training():
    print("\n========== TASK C: Training on Full Dataset ==========")
    
    # 1. Load the COMPLETE dataset
    print("Loading 50,000 training examples...")
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(50000, 10000)

    # 2. Configuration (Winning params from Task B)
    dims = [784, 40, 10] 
    epochs = 30
    lr = 0.1
    batch_size = 10

    print(f"Training network {dims} with LR={lr}...")
    
    # 3. Train
    net = Network(dims)
    _, _, _, _, test_acc = net.train(
        x_train, y_train, epochs, batch_size, lr, x_test, y_test
    )
    
    # 4. Analyze Results
    final_acc = test_acc[-1]
    print(f"\nFinal Test Accuracy: {final_acc:.2%}")
    
    if final_acc > 0.95:
        print("SUCCESS: The model passed the 95% accuracy threshold.")
    else:
        print("Result is slightly low. Re-running might help (random initialization).")

########################## Section D ##############################

def task_d_linear_classifier():
    print("\n========== TASK D: Linear Classifier ==========")
    
    # 1. Load the full dataset
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(50000, 10000)

    # 2. Configure the network (No hidden layers)
    dims = [784, 10] 
    net = Network(dims)
    
    # 3. Train the model
    print("Training linear classifier (30 epochs)...")
    _, _, _, train_acc, test_acc = net.train(
        x_train, y_train, 
        epochs=30, 
        batch_size=100, 
        learning_rate=0.1, 
        x_test=x_test, y_test=y_test
    )

    # 4. Plot Accuracy (Using Helper)
    plot_accuracy_comparison(train_acc, test_acc, "Task D: Linear Classifier Performance")

    # 5. Visualize the learned weights (Using Helper)
    print("Visualizing learned weights...")
    visualize_weights(net.parameters['W1'], "Visualizing Weights: What the Network 'Sees'")

########################## Section E ##############################

def task_e_bonus():
    print("\n========== TASK E: Creative Bonus (Deep Net + LR Decay) ==========")
    
    # 1. Load Data
    x_train, y_train, x_test, y_test = load_as_matrix_with_labels(50000, 10000)

    # IDEA 1: DEEP ARCHITECTURE (784 -> 128 -> 64 -> 10)
    dims = [784, 128, 64, 10]
    net = Network(dims)
    
    # IDEA 2: LEARNING RATE DECAY SCHEDULE
    schedule = [
        {"epochs": 20, "lr": 0.1,  "batch": 32},  # Stage 1: Fast Approach
        {"epochs": 15, "lr": 0.01, "batch": 32},  # Stage 2: Fine Tuning
        {"epochs": 15, "lr": 0.001, "batch": 32}  # Stage 3: Precision Settle
    ]
    
    full_test_acc = []
    full_train_acc = []
    
    print(f"Training Deep Network {dims} with Learning Rate Decay...")
    
    # We will track where the LR drops occur for the plot
    lr_drop_markers = []
    total_epochs_so_far = 0
    
    # Loop through our schedule
    for i, stage in enumerate(schedule):
        print(f"\n--- Stage {i+1}: LR={stage['lr']} for {stage['epochs']} epochs ---")
        
        # Track markers (except for the last stage)
        if i < len(schedule) - 1:
            total_epochs_so_far += stage['epochs']
            lr_drop_markers.append((total_epochs_so_far, f"LR Drop {i+1}"))

        # Train (continuing from previous state)
        _, _, _, train_acc, test_acc = net.train(
            x_train, y_train, 
            epochs=stage['epochs'], 
            batch_size=stage['batch'], 
            learning_rate=stage['lr'], 
            x_test=x_test, y_test=y_test
        )
        
        full_test_acc.extend(test_acc)
        full_train_acc.extend(train_acc)

    final_acc = full_test_acc[-1]
    print(f"\nFinal Test Accuracy: {final_acc:.2%}")
    
    # Plot using the new generic helper with vertical markers
    plot_accuracy_comparison(
        full_train_acc, 
        full_test_acc, 
        "Task E: Deep Network with Learning Rate Decay", 
        vlines=lr_drop_markers
    )

    if final_acc > 0.97:
        print("SUCCESS! The creative strategy worked.")

if __name__ == "__main__":
    #task_b_learning_rates()
    task_c_full_training()
    #task_d_linear_classifier()
    #task_e_bonus()