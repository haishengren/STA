import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split
from Model import smiles_to_index, Model


for i in range(11):

    df_train = pd.read_excel(r"enhanced_train_set_fold_{}.xlsx".format(i+1))
    df_test = pd.read_excel(r"enhanced_test_set_fold_{}.xlsx".format(i+1))

    y_train = df_train.iloc[:, 1]
    y_test = df_test.iloc[:, 1]

    smiles_train = df_train.iloc[:, 0].tolist()
    smiles_train = [i.replace('\xa0', '').replace(' ', '') for i in smiles_train]

    smiles_test = df_test.iloc[:, 0].tolist()
    smiles_test = [i.replace('\xa0', '').replace(' ', '') for i in smiles_test]

    # List of chemical elements in two characters
    special_elements = ['Li', 'Be', 'Na', 'Mg', 'Al', 'Si', 'Cl', 'Ca', 'Sc', 'Ti', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                        'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Rb', 'Sr', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag',
                        'Cd', 'In', 'Sn', 'Sb', 'Te', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                        'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb',
                        'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
                        'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'He', 'Ne', 'Ar', 'Kr',
                        'Xe']

    # Construct a lexicon of tokens derived from the database
    parsed_smiles = []
    smiles_all = smiles_train + smiles_test

    for smile_data in smiles_all:
        j = 0
        while j < len(smile_data):
            if j < len(smile_data) - 3 and smile_data[j:j + 4] == 'CIS-':
                parsed_smiles.append(smile_data[j:j + 4])
                j += 4
            if j < len(smile_data) - 1 and smile_data[j:j + 2] in special_elements:
                parsed_smiles.append(smile_data[j:j + 2])
                j += 2
            else:
                parsed_smiles.append(smile_data[j])
                j += 1

    char_to_idx = {'PAD': 0}
    char_to_idx.update({char: idx + 1 for idx, char in enumerate(sorted(set(parsed_smiles)))})

    vocabulary_size = len(char_to_idx)
    max_length = max(len(smiles) for smiles in smiles_all)
    print(f'vocabulary_size: {vocabulary_size}\nLength of longest SMILES: {max_length}\ntoken_to_index: {char_to_idx}')

    # Data processing and segmentation
    indices_train = torch.tensor(
        np.array([smiles_to_index(smiles, char_to_idx, max_length) for smiles in smiles_train]))
    indices_test = torch.tensor(np.array([smiles_to_index(smiles, char_to_idx, max_length) for smiles in smiles_test]))
    y_train = torch.tensor(y_train.values, dtype=torch.float)
    y_test = torch.tensor(y_test.values, dtype=torch.float)

    train_val_dataset = TensorDataset(indices_train, y_train)
    test_dataset = TensorDataset(indices_test, y_test)

    # Partition the dataset. 80% for training, 10% for validation, 10% for testing
    val_size = int(0.1 * len(train_val_dataset))
    train_size = len(train_val_dataset) - val_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    # create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=10)
    test_loader = DataLoader(test_dataset, batch_size=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model and train it with the GPU
    model = Model(vocab_size=vocabulary_size, length=max_length, d_model=512, num_heads=8, num_layers=12)
    model.to(device)
    print(model)

    # Instantiated optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    loss_fn = nn.L1Loss()

    # train the model and evaluation
    train_losses = []
    val_losses = []
    test_losses = []
    epochs_list = []

    patience = 5
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(100):

        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # Moving data to the GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # empty the gradient
            optimizer.zero_grad()

            # forward propagation
            outputs, _ = model(inputs)

            # calculate training loss
            loss = loss_fn(outputs.squeeze(), targets)  # 注意：outputs是形状为 (batch_size, 1) 的张量，目标 y 是形状为 (batch_size,) 的标量
            loss.backward()  # backward propagation
            optimizer.step()  # update parameters

            # cumulative loss
            running_loss += loss.item()

            # Print training progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                avg_batch_loss = running_loss / (batch_idx + 1)
                print(f"Epoch [{epoch + 1}/{50}], Batch [{batch_idx + 1}/{len(train_loader)}], "
                      f"Train Loss: {avg_batch_loss:.4f}")

        # Print the average training loss for the current epoch
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{50}], Train Loss: {avg_train_loss:.4f}")

        # evaluation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Moving data to the GPU
                inputs, targets = inputs.to(device), targets.to(device)

                # forward propagation
                outputs, _ = model(inputs)

                # calculate and cumulative validation loss
                loss = loss_fn(outputs.squeeze(), targets)
                val_loss += loss.item()

        # print the average validation loss for the current epoch
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch [{epoch + 1}/{50}], Validation Loss: {avg_val_loss:.4f}")

        # determining whether to stop early
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # preservation of optimal models
            torch.save(model.state_dict(), "Cp.pth")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            epochs_list.append(epoch + 1)
            break

        # Record the epoch of the actual training
        epochs_list.append(epoch + 1)

    # save the model
    torch.save(model.state_dict(), 'model.pth')

    # test the saved model
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    y_pre = []
    y_true = []
    test_results = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Moving data to the GPU
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward propagation
            outputs, _ = model(inputs)

            # Ensure outputs are not scalar after squeeze
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)  # Add a new dimension to avoid zero-dimensional tensor

            # Go to CPU and remove extra dimensions
            y_pre.append(outputs)
            y_true.append(targets)

    y_pre = torch.cat(y_pre, dim=0).flatten()
    y_true = torch.cat(y_true, dim=0).flatten()

    MAE = torch.mean(torch.abs(y_pre - y_true))

    # Calculate R2
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pre) ** 2)
    R2 = 1 - ss_residual / ss_total

    test_losses.append(MAE)

    print(f"Test MAE: {MAE:.4f}")
    print(f"Test R²: {R2:.4f}")