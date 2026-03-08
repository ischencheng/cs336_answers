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
- (a) It takes 340.9s and [9GB](./mprofile_20260308192718.dat). The longest token is [Ġaccomplishment](./TinyStoriesV2-GPT4-train_vocab.json) with  gpt2_bytes_to_unicode. 
- (b) The most time-consuming part is [pretokenization in subprocesses](./profile_output_2026-03-08T190641.txt)

Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
skip
