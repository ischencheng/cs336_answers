import json
import os
import sys

# Add project root to sys.path to import common testing utils
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from tests.common import gpt2_bytes_to_unicode

def find_longest_token(vocab_path):
    """
    Reads the vocabulary JSON file and finds the token with the maximum length.
    It also decodes the token back to its original readable string.
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
        
    longest_token_mapped_str = ""
    max_length = 0
    
    # Iterate over all tokens in the vocabulary
    for token_str in vocab.keys():
        # We can optionally skip special tokens like <|endoftext|> since they are manually added
        if token_str == "<|endoftext|>":
            continue
            
        # Since each character in the mapped string represents exactly 1 byte,
        # the length of the string is the exact byte length of the token.
        if len(token_str) > max_length:
            max_length = len(token_str)
            longest_token_mapped_str = token_str
            
    # Get the mapping to reconstruct the original bytes
    byte_encoder = gpt2_bytes_to_unicode()
    
    # Create a reverse mapping from visible character -> byte (integer 0-255)
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    
    # Convert the mapped string back into actual bytes
    token_bytes = bytes([byte_decoder[c] for c in longest_token_mapped_str])
    
    # Decode the bytes into a readable UTF-8 string
    # We use errors='replace' just in case the longest token is not a perfectly valid UTF-8 sequence
    decoded_token = token_bytes.decode('utf-8', errors='replace')
    
    print(f"Analysis for: {os.path.basename(vocab_path)}")
    print("-" * 50)
    print(f"Longest token length (in bytes) : {max_length}")
    # For Windows console printing compatibility with special characters
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        
    print(f"Mapped string representation: '{longest_token_mapped_str}'")
    print(f"Actual decoded string text      : '{decoded_token}'")

if __name__ == "__main__":
    # Analyze the training vocabulary by default
    vocab_file = "TinyStoriesV2-GPT4-train_vocab.json"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, vocab_file)
    
    if os.path.exists(full_path):
        find_longest_token(full_path)
    else:
        print(f"Error: Could not find vocab file at {full_path}")
