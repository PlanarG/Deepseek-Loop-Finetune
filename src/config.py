import configargparse

def parse_args():
    parser = configargparse.ArgumentParser(description='Finetune looped model on deepseek backbone')

    parser.add_argument('--config', is_config_file=True, help='Path to config file.')
    parser.add_argument('--max_data_length', type=int, default=256, help='Maximum data length for training.')
    parser.add_argument('--model_name', type=str, default='deepseek-ai/deepseek-llm-7b-base', help='Model name or path.')
    parser.add_argument('--dataset_name', type=str, default='meta-math/MetaMathQA', help='Dataset name or path.')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory for model and logs.')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--num_loops', type=int, default=2, help='Number of loops.')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for the optimizer.')
    parser.add_argument('--logging_steps', type=int, default=100, help='Number of steps between logging.')
    parser.add_argument('--save_steps', type=int, default=5, help='Number of epochs between model saves.')
    parser.add_argument('--eval_steps', type=int, default=1, help='Number of epochs between model evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='Warmup ratio for learning rate scheduler.')

    args = parser.parse_args()
    return args