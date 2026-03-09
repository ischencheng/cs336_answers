
from re import sub
import sys,os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from typing import Iterable, Iterator
import json
import regex as re
from cs336_basics.bbpe_tokenizer import PAT
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"



class Tokenizer:
    def __init__(self, vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], special_tokens:list[str] | None = None)->None:
        self.vocab=vocab
        self.merges=merges
        self.special_tokens = special_tokens or []
        self.special_pat = None
        if self.special_tokens:
            # Longest-first to avoid splitting overlapping special tokens.
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            self.special_pat = re.compile("(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")")
        self.vocab_inverse={v:k for k,v in vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

        return
    
    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None=None)->None:
        gpt2_byte_encoder=gpt2_bytes_to_unicode()
        gpt2_byte_decoder={v:k for k,v in gpt2_byte_encoder.items()}
        with open(vocab_filepath,'r') as vf:
            vocab_str_to_index=json.load(vf)
            vocab_index_to_bytes={i:bytes([gpt2_byte_decoder[c] for c in s]) for s,i in vocab_str_to_index.items()}

        if special_tokens:
            for special_token in special_tokens:
                special_token_bytes=special_token.encode('utf-8')
                if special_token_bytes not in set(vocab_index_to_bytes.values()):
                    vocab_index_to_bytes[len(vocab_index_to_bytes)]=special_token_bytes
        
        merges=[]
        with open(merges_filepath,'r') as mf:
            for line in mf.readlines():
                part1_str,part2_str=line.rstrip().split(' ')
                part1_bytes=bytes([gpt2_byte_decoder[c] for c in part1_str])
                part2_bytes=bytes([gpt2_byte_decoder[c] for c in part2_str])
                merges.append((part1_bytes,part2_bytes))
    
        return cls(vocab_index_to_bytes,merges,special_tokens)
    
    def _encode_pretoken_bytes(self, pre_token_bytes: bytes) -> list[bytes]:
        split = [bytes([b]) for b in pre_token_bytes]
        while len(split) > 1:
            best_idx = None
            best_rank = None
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_idx is None:
                break
            split[best_idx : best_idx + 2] = [split[best_idx] + split[best_idx + 1]]
        return split

    def encode(self,text:str)->list[int]:
        token_ids=[]
        chunks=[text]
        if self.special_pat:
            chunks=self.special_pat.split(text)
        for chunk in chunks:
            if self.special_tokens and chunk in self.special_tokens:
                token_ids.append(self.vocab_inverse[chunk.encode('utf-8')])
            else:
                for pre_token_str in PAT.findall(chunk):
                    token_bytes = self._encode_pretoken_bytes(pre_token_str.encode('utf-8'))
                    token_ids.extend(self.vocab_inverse[token_byte] for token_byte in token_bytes)
        return token_ids
                    
    def encode_iterable(self,iterable:Iterable[str])->Iterator[int]:
        for chunk in iterable:
            subchunks=[chunk]
            if self.special_pat:
                subchunks=self.special_pat.split(chunk)
            for subchunk in subchunks:
                if self.special_tokens and subchunk in self.special_tokens:
                    yield self.vocab_inverse[subchunk.encode('utf-8')]
                else:
                    for pre_token_str in PAT.findall(subchunk):
                        for token_byte in self._encode_pretoken_bytes(pre_token_str.encode('utf-8')):
                            yield self.vocab_inverse[token_byte]
                
    def decode(self, ids:list[int])->str:
        bs=b''.join(self.vocab[id] for id in ids)
        return bs.decode('utf-8', errors='replace')
            


if __name__=="__main__":
    tokenizer=Tokenizer.from_files(VOCAB_PATH,MERGES_PATH,special_tokens=["<|endoftext|>"])
    def test_roundtrip_unicode_string_with_special_tokens():
        test_string = "Héllò hôw <|endoftext|><|endoftext|> are ü? 🙃<|endoftext|>"
        encoded_ids = tokenizer.encode(test_string)
        tokenized_string = [tokenizer.decode([x]) for x in encoded_ids]
        # Ensure the special <|endoftext|> token is preserved
        assert tokenized_string.count("<|endoftext|>") == 3

        decoded_string = tokenizer.decode(encoded_ids)
        assert test_string == decoded_string

    test_roundtrip_unicode_string_with_special_tokens()