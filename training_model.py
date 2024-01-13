from paper_b_99_dl_bert_train_modularization import run_training


def gather_user_input():
    print("Enter single parameters if you want to run once; else, enter parameters separated by space.")
    grid_search_params = {}
    grid_search_params["LEARNING_RATE"] = [float(val) for val in
                                           input("Enter learning rate: ").split()]
    grid_search_params["BATCH_SIZE"] = [int(val) for val in
                                        input("Enter batch size: ").split()]
    grid_search_params["WARMUP_STEPS"] = [int(val) for val in
                                          input("Enter warmup steps: ").split()]
    grid_search_params["NUM_EPOCHS"] = [int(val) for val in
                                        input("Enter number of epochs: ").split()]
    grid_search_params["WEIGHT_DECAY"] = [float(val) for val in
                                          input("Enter weight decay: ").split()]
    grid_search_params["DROP_OUT_RATE"] = [float(val) for val in
                                           input("Enter dropout rate: ").split()]

    # Additional parameters for grid search
    print("Choose a BERT model:")
    print("1. bert-base-uncased")
    print("2. bert-large-uncased")
    print("3. bert-base-cased")

    model_choice = input("Enter the number of the BERT model you want to use: ")

    # Map user choice to the corresponding BERT model
    bert_models = {
        "1": "bert-base-uncased",
        "2": "bert-large-uncased",
        "3": "bert-base-cased"
    }

    selected_model = bert_models.get(model_choice)

    if selected_model:
        grid_search_params["bert_model"] = selected_model
    else:
        print("Invalid choice. Defaulting to bert-base-uncased.")
        grid_search_params["bert_model"] = "bert-base-uncased"  # Default to base if invalid choice

    grid_search_params["label_map"] = {"continue": 0, "not_continue": 1}
    grid_search_params["dataset_path"] = input("Enter dataset path e.g ("
                                               "shared_data/dataset_2_5_pair_sentences_reclass.jsonl): ")
    grid_search_params["confusion_matrix_name"] = input("Enter confusion matrix file name: ")
    grid_search_params["training_and_validation_losses"] = input("Enter training and validation losses file name: ")

    run_training(grid_search_params)


if __name__ == "__main__":
    gather_user_input()
