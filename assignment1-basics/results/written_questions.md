## 2.1 The Unicode Standard
Problem (unicode1): Understanding Unicode (1 point)
- (a) It returns`'\x00'`, the null character (often denoted as U+0000 or NULL).
- (b) `__repr__()` returns machine-readable string. For printable input object, it returns a string with single quotation mark to indicate its return is a literal. For nonprintable pbject input, it calls `ord()` to convert the `chr()` return string object to the *escape sequence* with single quotation mark. However, `print()` returns `None` so its return value doesn't trigger an additional `__repr__()`. In `print()`, it convert the input object to string object and write to the pipe. That is to say, `print()` writes human-readable string.
- (c)  The character exists in the string's memory as a valid data type but it is non-printable. Therefore, it is visible as an escape sequence in the shell 's echoing but results in novisible marks on the screen whne printed.

## 2.2 Unicode Encodings
Problem (unicode2): Unicode Encodings (3 points)
- (a) UTF-8 is the most memory efficient. For the most common english characters, it uses one byte per character. For chinese characters, it uses 3 bytes. It is length-variable. Instead UTF-32 is length-fixed. Every character occupies 4 bytes. Furthermore, UTF-16 and UTF-32 has many redundant null bytes and has the byte-ordering issues which implies the trained byte-level tokenizer may not work on a machine of different endiness.
- (b)  Not every single byte corresponds to a character and not every character corresponds to a single byte. For example, `b"\xe4\xbd\xa0"` corresponds to `'你'` However, 'utf-8' codec can't decode byte 0xe4 in position 0: unexpected end of data.
- (c) `b'\xff\xff'` the prefix of the bytes indicates the ordering but `b\xff` doesn't match any ordering.

## 2.5 Experimenting with BPE Tokenizer Training
Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
- (a) It takes 148.8s and [9GB](./mprofile_20260308192718.dat). The longest token is [Ġaccomplishment](./TinyStoriesV2-GPT4-train_vocab.json) with  gpt2_bytes_to_unicode. 
- (b) The most time-consuming part is to [get the most frequent pair](./profile_output_2026-03-09T193335.txt)

Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
skip

## 2.7 Experiments
Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
skip

## 3.6 The Full Transformer LM
Problem (transformer_accounting): Transformer LM resource accounting (5 points)
- (a) ~2.05e9 parameters (assuming weight tying for embedding and final linear layer). Memory: 2.05e9 * 4 bytes ≈ 7.64 GB.
    - `embedding`: d_model * vocab_size
    - `norm`: d_model
    - `transformer_blocks`: num_layers * (
        ffn: 3 * d_model * d_ff +
        rmsnorm: 2 * d_model +
        sdpa: 4 * d_model * d_model
      )
    - `linear`: d_model * vocab_size (shared with embedding)

- (b) ~4.52e12 FLOPs total.
    - **Matrix Multiplies (Dominant)**:
        - `QKV & O Projections`: num_layers * batch_size * context_length * (8 * d_model**2)
        - `FFN (SwiGLU)`: num_layers * batch_size * context_length * (6 * d_model * d_ff)
        - `Attention Scores & Output`: num_layers * batch_size * context_length * (4 * context_length * d_model)
        - `Final Linear`: 2 * batch_size * context_length * d_model * vocab_size
    - **Element-wise Operations (Minority)**:
        - `Transformer Blocks Element-wise`: num_layers * batch_size * context_length * (13 * d_model + 6 * context_length * num_heads)
        - `Final RMSNorm`: 2 * batch_size * context_length * d_model

- (c) If considering a single operation, the final linear layer (LM Head) requires the most FLOPs. However, collectively across the entire model, the Feed-Forward Networks (SwiGLU layers) consume the vast majority of the FLOPs (~67% of the total).

- (d) FLOPs Proportions by Model Size:
    - **GPT-2 Small** (12L, 768d): Total ~3.50e11 FLOPs
        - `FFN (SwiGLU)`: 49.75%
        - `QKV & O Projections`: 16.58%
        - `Attention Scores & Output`: 11.06%
        - `Final Linear Head`: 22.61%
    - **GPT-2 Medium** (24L, 1024d): Total ~1.03e12 FLOPs
        - `FFN (SwiGLU)`: 59.87%
        - `QKV & O Projections`: 19.96%
        - `Attention Scores & Output`: 9.98%
        - `Final Linear Head`: 10.20%
    - **GPT-2 Large** (36L, 1280d): Total ~2.26e12 FLOPs
        - `FFN (SwiGLU)`: 64.20%
        - `QKV & O Projections`: 21.40%
        - `Attention Scores & Output`: 8.56%
        - `Final Linear Head`: 5.84%
    - **Conclusion**: As the model size increases (specifically `d_model` and `num_layers`), the FFN and Projections (which scale with $d_{model}^2$) take up an increasingly dominant proportion of the total FLOPs. Conversely, the final linear head (which is bounded by the fixed `vocab_size`) and the attention mechanism (which scales linearly with $d_{model}$ for a fixed sequence length) become proportionally much smaller.

- (e) Context length 16,384 for GPT-2 XL:
    - The total FLOPs increases dramatically from ~4.51e12 to ~1.50e14. Because the self-attention operation scales quadratically with context length ($O(T^2)$), its relative contribution explodes to dominate the compute, increasing from ~7.14% (at context length 1,024) to ~55.15% (at context length 16,384) of total FLOPs, overshadowing the FFN layers.