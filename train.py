"""
Federated learning training script for fine-tuning language models.
"""

import os
import random
from typing import List, Dict
import fire
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict
)
import datasets
from fed_utils import (
    fedavg,
    hetlora, 
    load_hetlora_weights, 
    select_clients, 
    evaluate_global_model, 
    FederatedClient,
    Prompter
)

# Reduce verbosity of datasets library
datasets.utils.logging.set_verbosity_error()


def fl_finetune(
        exp_name: str = 'fedavg-1B',
        # Model/data params
        base_model: str = 'meta-llama/Llama-3.2-1B',
        data_path: str = './data',
        model_dir: str = './models',
        # FL hyperparams
        client_selection_strategy: str = 'random',
        client_selection_frac: float = 0.1,
        num_communication_rounds: int = 20,
        num_clients: int = 8,
        # Federation mode
        federation_mode: str = "homo",  # "none", "homo", "hetero", or "seq"
        # Local training hyperparams
        local_batch_size: int = 128,
        local_micro_batch_size: int = 8,
        local_num_epochs: int = 3,
        local_learning_rate: float = 3e-4,
        local_val_set_size: int = 0,
        val_data_path: str = "",
        cutoff_len: int = 512,
        # LoRA hyperparams
        lora_r: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: List[str] = [
            "q_proj",
            "v_proj",
        ],
        # LLM hyperparams
        train_on_inputs: bool = False,
        group_by_length: bool = False
):
    """
    Train a language model using federated learning.
    
    Args:
        exp_name: Experiment name
        base_model: Base model to fine-tune
        data_path: Path to training data
        model_dir: Directory to save outputs
        client_selection_strategy: Strategy for selecting clients
        client_selection_frac: Fraction of clients to select
        num_communication_rounds: Number of communication rounds
        num_clients: Total number of clients
        federation_mode: Federation strategy ("none", "homo", "hetero", or "seq")
        local_batch_size: Batch size for local training
        local_micro_batch_size: Micro batch size for gradient accumulation
        local_num_epochs: Number of epochs for local training
        local_learning_rate: Learning rate for local training
        local_val_set_size: Size of validation set (0 for no validation)
        val_data_path: Path to validation data
        cutoff_len: Maximum sequence length
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        train_on_inputs: Whether to train on inputs
        group_by_length: Whether to group sequences by length
    """
    # Validate federation mode
    assert federation_mode in ["none", "homo", "hetero", "seq"], \
        "federation_mode must be one of 'none', 'homo', 'hetero', or 'seq'"

    use_hetlora = (federation_mode == "hetero")
    # Federation includes homo and hetero modes. Seq and none are non-federated.
    use_federation = (federation_mode in ["homo", "hetero"])
    use_sequential = (federation_mode == "seq")

    # Create experiment output directory
    model_output_dir = os.path.join(model_dir, exp_name)
    os.makedirs(model_output_dir, exist_ok=True)

    # Log experiment parameters
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Federated Fine-tuning with:\n"
            f"Experiment: {exp_name}\n"
            f"Model: {base_model}\n"
            f"Data: {data_path}\n"
            f"Output: {model_output_dir}\n"
            f"Federation mode: {federation_mode}\n"
            f"Clients: {num_clients} (selection: {client_selection_frac:.2f} using {client_selection_strategy})\n"
            f"Communication rounds: {num_communication_rounds}\n"
            f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}\n"
        )

    # Verify model and data paths
    assert base_model, "Please specify a base_model"
    client_data_path = os.path.join(data_path, str(num_clients))
    assert os.path.exists(client_data_path), f"Data directory {client_data_path} not found"

    # Set up DDP if needed
    gradient_accumulation_steps = local_batch_size // local_micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
    else:
        device_map = "auto"

    # Initialize tokenizer and prompter
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    prompter = Prompter()

    # Load the base model with quantization
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    # Prepare model for training
    base_model_obj = prepare_model_for_kbit_training(base_model_obj)

    # Tokenization function for data preprocessing
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()
        return result

    # Data preprocessing function
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"] if 'input' in data_point.keys() else None,
            data_point["output"],
        )
        tokenized_full_prompt = tokenize(full_prompt)
        
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], 
                data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]
        return tokenized_full_prompt

    # Prepare client LoRA ranks for HetLoRA
    client_lora_ranks = {}
    if use_hetlora:
        ranks = [4, 8, 16]
        num_rank_categories = len(ranks)
        base_count = num_clients // num_rank_categories
        remainder = num_clients % num_rank_categories

        counts = [base_count + 1 if i < remainder else base_count
                for i in range(num_rank_categories)]
        rank_assignments = [rank for i, rank in enumerate(ranks)
                             for _ in range(counts[i])]
        # rank_assignments = [4, 8, 8, 8, 8, 8, 8, 8]
        
        random.seed(309)  # For reproducibility
        random.shuffle(rank_assignments)
        
        client_lora_ranks = dict(enumerate(rank_assignments))
        print("Using HetLoRA with client ranks:", client_lora_ranks)
        rank_distribution = {}
        for rank in client_lora_ranks.values():
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        print("Rank distribution:", rank_distribution)
        
        # Global rank will be the maximum rank
        global_rank = max(client_lora_ranks.values())
        print(f"Global rank set to {global_rank} (max of all client ranks)")
    else:
        # Use the same rank for all clients when using FedAvg or no federation
        global_rank = lora_r
        for client_id in range(num_clients):
            client_lora_ranks[client_id] = lora_r
        print(f"Using homogeneous LoRA with rank {lora_r} for all clients")

    # Initialize global parameters (or initial parameters for seq mode)
    # Create a global config for reference (used for homo, hetero, and seq initial/saving)
    global_config = LoraConfig(
        r=global_rank, # In seq mode, global_rank is just lora_r
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Initialize model with adapter and get initial parameters
    initial_model = get_peft_model(base_model_obj, global_config)
    current_params = get_peft_model_state_dict(initial_model)
    
    # Now remove the adapter from the base model object for client loading
    base_model_obj = initial_model.get_base_model()
    print(type(initial_model), type(base_model_obj))
    del initial_model  # Free up memory

    # Start training
    print(f"Starting {'federated' if use_federation else ('sequential' if use_sequential else 'isolated')} training...")
    
    # Initialize tracking variables
    previously_selected_clients = set()
    last_client_id = None
    local_dataset_len_dict = {}

    # Determine the number of loops/epochs
    # For seq mode, num_communication_rounds can mean number of passes over all clients
    # For simplicity here, let's assume 1 pass if seq mode is chosen, unless num_communication_rounds > 1
    # If rounds > 1 in seq mode, the state passes from the last client of round N to the first client of round N+1
    num_training_loops = num_communication_rounds
    if use_sequential:
        print("Sequential mode selected. Forcing num_training_loops to 1.")
        num_training_loops = 1

    # Main training loop
    for epoch in tqdm(range(num_training_loops), desc="Training Loop/Rounds"):
        print(f"\nLoop/Round {epoch+1}/{num_training_loops}")
        
        # Select clients for this round/loop
        if use_sequential:
            # Iterate through all clients sequentially
            selected_clients = list(range(num_clients))
            print(f"Sequential mode: Processing clients {selected_clients}")
        elif use_federation:
            # Select a fraction of clients for federated learning
            selected_clients = select_clients(
                num_clients, 
                client_selection_frac, 
                client_selection_strategy,
                seed=epoch # Use epoch as seed for reproducibility per round
            )
            print(f"Federated mode: Selected clients {selected_clients}")
        else: # mode == "none"
            # Train all clients independently (similar structure to sequential but without state passing)
            selected_clients = list(range(num_clients))
            print(f"Isolated mode: Processing clients {selected_clients}")

        client_states_for_aggregation = {} # Only used in federated modes

        # Train each selected client
        for client_id in selected_clients:
            # Get client-specific rank (always lora_r for homo and seq)
            client_rank = client_lora_ranks[client_id] # This dict holds lora_r for seq/homo
            print(f"\nClient {client_id}: Using LoRA rank {client_rank}")
            
            # Create client-specific configuration
            client_config = LoraConfig(
                r=client_rank,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            # Create a fresh client model by adding adapter to the base model
            # Ensure base_model_obj has no PEFT adapters before adding a new one
            if hasattr(base_model_obj, "get_base_model"):
                # This check might be redundant if we manage base_model_obj correctly, but safe to keep
                print('\nEnsuring clean base model before adding adapter\n')
                base_model_obj = base_model_obj.get_base_model()
                
            client_model = get_peft_model(
                base_model_obj, 
                client_config,
                adapter_name="default" # Use a consistent adapter name
            )
            
            # Load weights into the client model
            if use_federation:
                if use_hetlora:
                    print(f"Client {client_id}: Loading weights from global parameters (heterogeneous)")
                    client_weights = load_hetlora_weights(
                        client_config,
                        current_params, # Use current_params which holds global state
                        client_rank
                    )
                    set_peft_model_state_dict(client_model, client_weights, "default")
                else: # Homo FedAvg
                    print(f"Client {client_id}: Loading weights from global parameters (homogeneous)")
                    set_peft_model_state_dict(client_model, current_params, "default")
            elif use_sequential:
                # Load the state from the *previous* step (or initial if client_id == 0 and epoch == 0)
                print(f"Client {client_id}: Loading weights from previous step (sequential)")
                # current_params holds the state from the previous client or the initial state
                set_peft_model_state_dict(client_model, current_params, "default")
            # else: mode == "none", start fresh from base + new adapter (no loading needed)

            # Initialize client trainer
            client = FederatedClient(client_id, client_model, client_data_path, model_output_dir)
            
            # Prepare client for training
            print(f"Preparing client {client_id} for training")
            client.prepare_dataset(generate_and_tokenize_prompt, local_val_set_size)
            client.build_trainer(
                tokenizer,
                local_micro_batch_size,
                gradient_accumulation_steps,
                local_num_epochs,
                local_learning_rate,
                group_by_length,
                ddp
            )

            # Train client
            print(f"Training client {client_id}")
            client.train()

            # Get the trained parameters from the client model
            trained_client_params = get_peft_model_state_dict(client_model)

            # Save client model weights (as done by client.save_client_state)
            # We manually replicate the saving part here to ensure the correct state is saved
            print(f"Saving model state for client {client_id} after training")
            local_dataset_len_dict[client_id] = len(client.local_train_dataset) # Get dataset length
            participating_clients = previously_selected_clients | {client_id} # Update participants (less relevant for seq)
            last_client_id = client_id # Track last client processed

            # Create client directory within the current epoch/loop directory
            client_epoch_dir = os.path.join(model_output_dir, str(epoch), f"client_{client_id}")
            os.makedirs(client_epoch_dir, exist_ok=True)
            
            # Save adapter weights
            adapter_weights_path = os.path.join(client_epoch_dir, "adapter_model.bin")
            torch.save(trained_client_params, adapter_weights_path)
            print(f"Saved adapter weights to {adapter_weights_path}")
            
            # Save client-specific config (which is the same as global_config in seq/homo mode)
            client_config.save_pretrained(client_epoch_dir)
            print(f"Saved adapter config to {client_epoch_dir}")

            # Update current_params for the next step in sequential mode
            if use_sequential:
                print(f"Updating current parameters with client {client_id}'s result.")
                current_params = trained_client_params # Pass the trained state to the next client
            
            # Store client state for potential aggregation (only used in federated modes)
            if use_federation:
                client_states_for_aggregation[client_id] = trained_client_params

            # Get the base model back for the next client
            # This ensures we always add a fresh adapter to the clean base model
            base_model_obj = client_model.get_base_model()
            
            # Clean up client-related objects
            del client
            del client_model
            del trained_client_params # Free memory
            if use_federation and use_hetlora:
                del client_weights # Free memory if HetLoRA was used

        # Perform model aggregation if using federation
        if use_federation:
            print("\n======= Aggregating client models =======")
            # Note: We need the saved client models for aggregation,
            # The current implementation of fedavg/hetlora reads from disk based on epoch/client_id.
            # We need to ensure the parameters passed to fedavg/hetlora are correct.
            # Let's pass the global state `current_params` and let the functions update it.
            
            # Get the list of clients that participated in *this* round
            clients_in_this_round = selected_clients # Use the list of clients selected for this round

            if use_hetlora:
                print("Using HetLoRA sparsity-weighted aggregation")
                current_params = hetlora(
                    current_params, # Pass the current global state
                    clients_in_this_round,
                    model_output_dir,
                    local_dataset_len_dict,
                    epoch,
                    client_lora_ranks
                )
            else: # Homo FedAvg
                print("Using FedAvg homogeneous aggregation")
                current_params = fedavg(
                    current_params, # Pass the current global state
                    clients_in_this_round,
                    model_output_dir,
                    local_dataset_len_dict,
                    epoch
                )
            
            # Save aggregated global model parameters
            round_dir = os.path.join(model_output_dir, str(epoch))
            # The directory should already exist from client saving, but make sure
            os.makedirs(round_dir, exist_ok=True) 
            global_params_path = os.path.join(round_dir, "global_adapter_model.bin")
            torch.save(current_params, global_params_path)
            print(f"Saved aggregated global parameters to {global_params_path}")
            
            # Save global config (should be consistent across rounds)
            config_path = os.path.join(round_dir) # Save to the round directory
            global_config.save_pretrained(config_path)
            print(f"Saved global config with rank {global_rank} to {config_path}")
        
        elif use_sequential:
            print("\nSequential mode: No aggregation performed. State passed to next client.")
            # Optionally save the final state after the last client in the sequence for this epoch
            final_seq_state_path = os.path.join(model_output_dir, str(epoch), "final_sequential_adapter_model.bin")
            torch.save(current_params, final_seq_state_path)
            global_config.save_pretrained(os.path.join(model_output_dir, str(epoch))) # Save config too
            print(f"Saved final sequential state for epoch {epoch} to {final_seq_state_path}")

        else: # mode == "none"
            print("\nIsolated mode: No aggregation performed. Client models saved individually.")

        # Evaluate global model (or current sequential model) if validation data is provided
        if val_data_path:
            try:
                print("\nEvaluating model state on validation data...")
                # Ensure base_model_obj is clean
                if hasattr(base_model_obj, "get_base_model"):
                    base_model_obj = base_model_obj.get_base_model()
                
                # Create a temporary model with the current parameters for evaluation
                # Use global_config as it defines the structure (rank might differ for HetLoRA, but eval uses max rank)
                eval_model = get_peft_model(base_model_obj, global_config)
                
                # Load the current parameters (either aggregated global or final sequential state)
                set_peft_model_state_dict(eval_model, current_params, "default")
                
                # Evaluate
                eval_loss = evaluate_global_model(
                    eval_model, 
                    val_data_path, 
                    generate_and_tokenize_prompt, 
                    batch_size=1, # Keep batch size small for evaluation consistency
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                print(f"Loop/Round {epoch + 1} validation loss: {eval_loss}")
                
                # Get base model back
                base_model_obj = eval_model.get_base_model()
                
                # Clean up
                del eval_model
            except Exception as e:
                print(f"Evaluation error: {e}")

    print(f"Training completed. Models saved to {model_output_dir}")


if __name__ == "__main__":
    fire.Fire(fl_finetune)