"""
Federated Learning Model for Viewability Prediction of Ads
This is part of AdFL paper
Modular implementation supporting 5, 10, 50, 100, or 500 users
All features have been obfuscated to protect privacy
Supports Differential Privacy for enhanced privacy protection
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import logging
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Differential Privacy imports (optional)
try:
    from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
    DP_AVAILABLE = True
except ImportError:
    DP_AVAILABLE = False
    logging.warning("TensorFlow Privacy not available. Install with: pip install tensorflow-privacy")

# Obfuscated feature names
LABEL_COL = 'target'

# Binary features (4 features)
bin_cols = ['bin_1', 'bin_2', 'bin_3', 'bin_4']

# Numeric features (14 features)
num_cols = [
    'num_1', 'num_2', 'num_3', 'num_4', 'num_5', 'num_6', 'num_7',
    'num_8', 'num_9', 'num_10', 'num_11', 'num_12', 'num_13', 'num_14'
]

# Categorical features (9 features)
cat_cols = [
    'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5',
    'cat_6', 'cat_7', 'cat_8', 'cat_9'
]

# Hash bucket sizes for categorical hashing
hash_bins = {
    "cat_1": 8,
    "cat_2": 8,
    "cat_3": 8,
    "cat_4": 8,
    "cat_5": 32,
    "cat_6": 256,
    "cat_7": 8,
    "cat_8": 16,
    "cat_9": 512,
}

# Model parameters
binary_input_shape = len(bin_cols)
numerical_input_shape = len(num_cols)
initial_bias_v = 0.5

# Configuration for different user counts
USER_CONFIGS = {
    2: {
        'early_stopping_patience': 2,
        'early_stopping_start_round': 2,
        'min_samples': 100,
        'max_samples': 400,
    },
    5: {
        'early_stopping_patience': 3,
        'early_stopping_start_round': 3,
        'min_samples': 100,
        'max_samples': 400,
    },
    10: {
        'early_stopping_patience': 5,
        'early_stopping_start_round': 5,
        'min_samples': 100,
        'max_samples': 500,
    },
    50: {
        'early_stopping_patience': 20,
        'early_stopping_start_round': 50,
        'min_samples': 100,
        'max_samples': 400,
    },
    100: {
        'early_stopping_patience': 30,
        'early_stopping_start_round': 100,
        'min_samples': 95,
        'max_samples': 400,
    },
    500: {
        'early_stopping_patience': 40,
        'early_stopping_start_round': 200,
        'min_samples': 50,
        'max_samples': 400,
    }
}

# Setup logging
log_filename = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def _validate_no_label_in_features() -> None:
    """Ensure label is not included in feature columns"""
    feature_cols = set(bin_cols) | set(num_cols) | set(cat_cols)
    if LABEL_COL in feature_cols:
        raise ValueError(f"Label leakage: {LABEL_COL!r} is included in feature columns")

_validate_no_label_in_features()

def get_model():
    """Build the federated learning model with three input groups"""
    from tensorflow.keras import layers
    
    # Categorical inputs and embeddings
    cat_inputs = []
    encoded_features = []

    for feature in cat_cols:
        inp = layers.Input(shape=(1,), name=feature, dtype=tf.string)
        cat_inputs.append(inp)
        hashed = layers.Hashing(num_bins=hash_bins.get(feature, 64), output_mode='int')(inp)
        emb_dim = min(50, hash_bins.get(feature, 64) // 2)
        emb = layers.Embedding(input_dim=hash_bins.get(feature, 64), output_dim=emb_dim)(hashed)
        emb = layers.Flatten()(emb)
        encoded_features.append(emb)
    
    # Numeric and binary inputs
    num_input = layers.Input(shape=(numerical_input_shape,), name='numeric_input')
    bin_input = layers.Input(shape=(binary_input_shape,), name='binary_input')
    
    # Project numeric and binary inputs
    num_dense = layers.Dense(64, activation='relu')(num_input)
    bin_dense = layers.Dense(200, activation='relu')(bin_input)
    merged = layers.Concatenate()(encoded_features + [num_dense, bin_dense])
    
    # Deep neural network layers
    hidden_layer_1 = layers.Dense(500, activation='relu')(merged)
    hidden_layer_2 = layers.Dense(250, activation='relu')(hidden_layer_1)
    hidden_layer_3 = layers.Dense(100, activation='relu')(hidden_layer_2)
    hidden_layer_4 = layers.Dense(50, activation='relu')(hidden_layer_3)
    hidden_layer_5 = layers.Dense(30, activation='relu')(hidden_layer_4)
    
    output_bias_v = tf.keras.initializers.Constant(initial_bias_v)
    output = layers.Dense(1, activation='sigmoid', name='target', bias_initializer=output_bias_v)(hidden_layer_5)
    
    model_inputs = cat_inputs + [num_input, bin_input]
    model = tf.keras.Model(inputs=model_inputs, outputs=[output])
    return model

def df_to_dataset(df, scaler=None, batch_size=32, shuffle=False):
    """Convert DataFrame to tf.data.Dataset"""
    df = df.copy()
    labels = df[LABEL_COL].values
    
    # Categorical features as string arrays
    cat_dict = {f: df[f].astype(str).values for f in cat_cols}
    
    # Numeric features with scaling
    numeric_arr = df[num_cols].values.astype(np.float32)
    if scaler is not None:
        numeric_arr = scaler.transform(numeric_arr)
    
    # Binary features
    binary_arr = df[bin_cols].values.astype(np.float32)
    
    # Combine into inputs dict
    inputs = {}
    inputs.update(cat_dict)
    inputs['numeric_input'] = numeric_arr
    inputs['binary_input'] = binary_arr
    
    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(10000)
    ds = ds.batch(batch_size)
    return ds

def split_dataset(df, batch_size=32, train_size=0.8, val_size=0.1):
    """Split dataset and fit scaler on training data"""
    
    train_df, temp_df = train_test_split(df, train_size=train_size, shuffle=False)
    val_df, test_df = train_test_split(temp_df, test_size=val_size / (1 - train_size), shuffle=False)
    
    scaler = MinMaxScaler()
    scaler.fit(train_df[num_cols].values.astype(np.float32))
    
    train_ds = df_to_dataset(train_df, scaler=scaler, batch_size=batch_size, shuffle=False)
    val_ds = df_to_dataset(val_df, scaler=scaler, batch_size=batch_size, shuffle=False)
    test_ds = df_to_dataset(test_df, scaler=scaler, batch_size=batch_size, shuffle=False)
    
    return train_ds, val_ds, test_ds

def combine_datasets(*datasets):
    """Combine multiple datasets"""
    combined = datasets[0]
    for ds in datasets[1:]:
        combined = combined.concatenate(ds)
    return combined

class Client:
    """Federated learning client with optional differential privacy"""
    def __init__(self, client_id, train_dataset, val_dataset, test_dataset, 
                 use_dp=False, l2_norm_clip=1.0, noise_multiplier=0.1, num_microbatches=1):
        self.client_id = client_id
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.use_dp = use_dp
        self.model = get_model()
        
        # Configure optimizer based on DP setting
        if use_dp and DP_AVAILABLE:
            logger.info(f"Client {client_id}: Using DP-SGD optimizer (clip={l2_norm_clip}, noise={noise_multiplier})")
            self.optimizer = DPKerasAdamOptimizer(
                l2_norm_clip=l2_norm_clip,
                noise_multiplier=noise_multiplier,
                num_microbatches=num_microbatches,
                learning_rate=0.001
            )
        else:
            if use_dp and not DP_AVAILABLE:
                logger.warning(f"Client {client_id}: DP requested but not available, using standard Adam")
            self.optimizer = tf.keras.optimizers.Adam()
        
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
        )

    def train(self):
        """Train on local data"""
        # Use model.fit() for DP optimizer compatibility
        if self.use_dp and DP_AVAILABLE:
            # DP optimizer requires using model.fit() or optimizer._compute_gradients()
            self.model.fit(self.train_dataset, epochs=1, verbose=0)
        else:
            # Standard training loop
            self._train_standard()
    
    @tf.function
    def _train_standard(self):
        """Standard training without DP"""
        for batch in self.train_dataset:
            with tf.GradientTape() as tape:
                predictions = self.model(batch[0], training=True)
                loss = self.model.compiled_loss(batch[1], predictions)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def reset_metrics(self):
        for metric in self.model.metrics:
            if hasattr(metric, 'reset_states'):
                metric.reset_states()
            elif hasattr(metric, 'reset_state'):
                metric.reset_state()

    def evaluate_val(self):
        return self.model.evaluate(self.val_dataset, verbose=0)[0]

    def evaluate_test(self):
        return self.model.evaluate(self.test_dataset, verbose=0)

def train_federated_model(client_data, num_users, rounds=1000, experiment_id=0, 
                         use_dp=False, l2_norm_clip=1.0, noise_multiplier=0.1, num_microbatches=1):
    """
    Train federated learning model with optional differential privacy
    
    Args:
        client_data: Dictionary of client datasets
        num_users: Number of users (5, 10, 50, 100, or 500)
        rounds: Maximum number of training rounds
        experiment_id: Experiment identifier for output files
        use_dp: Enable differential privacy (requires tensorflow-privacy)
        l2_norm_clip: L2 norm clipping threshold for DP
        noise_multiplier: Noise multiplier for DP
        num_microbatches: Number of microbatches for DP
    """
    config = USER_CONFIGS[num_users]
    
    if use_dp:
        if not DP_AVAILABLE:
            logger.error("Differential Privacy requested but tensorflow-privacy not installed!")
            logger.error("Install with: pip install tensorflow-privacy")
            raise ImportError("tensorflow-privacy required for DP mode")
        logger.info(f"Differential Privacy ENABLED: clip={l2_norm_clip}, noise={noise_multiplier}")
    else:
        logger.info("Differential Privacy DISABLED")
    
    clients = [
        Client(
            client_id=i,
            train_dataset=client_data[i]['train'],
            val_dataset=client_data[i]['val'],
            test_dataset=client_data[i]['test'],
            use_dp=use_dp,
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches
        )
        for i in range(1, len(client_data) + 1)
    ]
    
    global_model = get_model()
    global_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()]
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    client_metrics = {i: {'loss': [], 'accuracy': [], 'auc': []} for i in range(1, len(client_data) + 1)}
    global_metrics = {'loss': [], 'accuracy': [], 'auc': []}
    
    logger.info(f"Starting federated training for {len(clients)} clients, {rounds} total rounds")
    logger.info(f"Configuration: {config}")
    
    for round_num in range(1, rounds + 1):
        logger.info(f"=== Round {round_num}/{rounds} ===")
        client_weights = []

        # Train each client
        for client in clients:
            logger.debug(f"Training client {client.client_id}...")
            client.reset_metrics()
            client.train()
            client_weights.append(client.get_weights())
            logger.debug(f"Client {client.client_id} finished training")
        
        # Average weights (federated averaging)
        new_weights = []
        for weights_list_tuple in zip(*client_weights):
            averaged_weights = np.mean(weights_list_tuple, axis=0)
            new_weights.append(averaged_weights)
        
        global_model.set_weights(new_weights)
        for client in clients:
            client.set_weights(new_weights)
        
        # Reset global model metrics
        for metric in global_model.metrics:
            if hasattr(metric, 'reset_states'):
                metric.reset_states()
            elif hasattr(metric, 'reset_state'):
                metric.reset_state()
        
        # Validation
        logger.info("Evaluating clients on validation splits...")
        val_losses = [client.evaluate_val() for client in clients]
        avg_val_loss = np.mean(val_losses)

        # Test evaluation
        logger.info("Evaluating clients on test splits...")
        for client in clients:
            test_metrics = client.evaluate_test()
            client_metrics[client.client_id]['loss'].append(test_metrics[0])
            client_metrics[client.client_id]['accuracy'].append(test_metrics[1])
            client_metrics[client.client_id]['auc'].append(test_metrics[2])
        
        # Global test evaluation
        combined_test_ds = combine_datasets(*[data['test'] for data in client_data.values()])
        global_test_metrics = global_model.evaluate(combined_test_ds, verbose=0)
        global_metrics['loss'].append(global_test_metrics[0])
        global_metrics['accuracy'].append(global_test_metrics[1])
        global_metrics['auc'].append(global_test_metrics[2])
        
        logger.info(f'Round {round_num:2d}, Average Validation Loss={avg_val_loss:.4f}')
        
        # Early stopping
        if round_num >= config['early_stopping_start_round']:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                logger.info(f"New best validation loss: {best_val_loss:.4f}. Resetting patience counter.")
            else:
                patience_counter += 1
                logger.info(f"No improvement. Patience: {patience_counter}/{config['early_stopping_patience']}")
                if patience_counter >= config['early_stopping_patience']:
                    logger.info("Early stopping triggered.")
                    break
        else:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        logger.info(f'Round {round_num:2d}, Global Metrics - Loss: {global_metrics["loss"][-1]:.4f}, '
                   f'Accuracy: {global_metrics["accuracy"][-1]:.4f}, AUC: {global_metrics["auc"][-1]:.4f}')

    # Save final results
    result = (f'Round {round_num:2d}, Global Model Metrics - '
             f'Loss: {global_metrics["loss"][-1]:.4f}, '
             f'Accuracy: {global_metrics["accuracy"][-1]:.4f}, '
             f'AUC: {global_metrics["auc"][-1]:.4f}')
    
    output_file = f'{num_users}users_experiment_{experiment_id}.txt'
    with open(output_file, 'w') as file:
        file.write(result)
    
    logger.info(f"Results saved to {output_file}")
    return global_metrics

def prepare_client_data(df, num_users, batch_size=32, user_id_col='user_id'):
    """
    Prepare client datasets from dataframe
    
    Args:
        df: Input dataframe with obfuscated features
        num_users: Number of users to select (5, 10, 50, 100, or 500)
        batch_size: Batch size for training
        user_id_col: Column name for user identifier (if None, creates sequential splits)
    
    Returns:
        Dictionary of client datasets
    """
    config = USER_CONFIGS[num_users]
    
    # Original logic with user_id column
    # Filter users by sample count
    user_counts = df[user_id_col].value_counts()
    filtered_user_counts = user_counts[
        (user_counts >= config['min_samples']) & 
        (user_counts <= config['max_samples'])
    ]
    filtered_df = df[df[user_id_col].isin(filtered_user_counts.index)]
    
    # Select top N users by data volume
    user_counts_filtered = filtered_df[user_id_col].value_counts()
    selected_users = user_counts_filtered.head(num_users).index.tolist()
    
    logger.info(f"Selected top {num_users} users - data per user:")
    logger.info("\n" + user_counts_filtered.head(num_users).to_string())
    logger.info(f"Total data points: {user_counts_filtered.head(num_users).sum()}")
    
    # Create client datasets
    client_data = {}
    for i, user_id in enumerate(selected_users, start=1):
        user_df = filtered_df[filtered_df[user_id_col] == user_id].copy()
        train_ds, val_ds, test_ds = split_dataset(user_df, batch_size=batch_size)
        client_data[i] = {'train': train_ds, 'val': val_ds, 'test': test_ds}
    
    return client_data

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Federated Learning Model for Viewability Prediction (part of AdFL paper) of Ads to simulate running within the browser.')
    parser.add_argument('--num_users', type=int, choices=[2, 5, 10, 50, 100, 500], default=5,
                       help='Number of users (2, 5, 10, 50, 100, or 500) - default: 5')
    parser.add_argument('--data_file', type=str, default='data_sample.csv',
                       help='Path to obfuscated data file')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Maximum number of training rounds - default: 5')
    parser.add_argument('--experiments', type=int, default=1,
                       help='Number of experiments to run')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    
    # Differential Privacy arguments
    parser.add_argument('--use_dp', action='store_true',
                       help='Enable differential privacy (requires tensorflow-privacy)')
    parser.add_argument('--l2_norm_clip', type=float, default=1.0,
                       help='L2 norm clipping threshold for DP (default: 1.0)')
    parser.add_argument('--noise_multiplier', type=float, default=0.1,
                       help='Noise multiplier for DP (default: 0.1, higher=more privacy)')
    parser.add_argument('--num_microbatches', type=int, default=1,
                       help='Number of microbatches for DP (default: 1)')
    
    args = parser.parse_args()
    
    # Warn if configuration requires more data than available in sample
    if args.num_users >= 50 and args.data_file == 'data_sample.csv':
        logger.warning("="*80)
        logger.warning(f"WARNING: Requested {args.num_users} users but data_sample.csv only contains 2,000 records (10 users).")
        logger.warning(f"Configurations with 50, 100, or 500 users require a larger dataset.")
        logger.warning(f"This will fail or produce invalid results. Use --data_file to specify a larger dataset.")
        logger.warning(f"Recommended: Use 2, 5, or 10 users with the provided sample data.")
        logger.warning("="*80)
    
    logger.info(f"Loading data from {args.data_file}")
    df = pd.read_csv(args.data_file)
    
    logger.info(f"Preparing client data for {args.num_users} users")
    client_data = prepare_client_data(df, args.num_users, args.batch_size)
    
    logger.info(f"Running {args.experiments} experiments")
    for exp_id in range(args.experiments):
        logger.info(f"\n{'='*60}\nExperiment {exp_id + 1}/{args.experiments}\n{'='*60}")
        train_federated_model(
            client_data, 
            args.num_users, 
            args.rounds, 
            exp_id,
            use_dp=args.use_dp,
            l2_norm_clip=args.l2_norm_clip,
            noise_multiplier=args.noise_multiplier,
            num_microbatches=args.num_microbatches
        )

if __name__ == '__main__':
    main()
