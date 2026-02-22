from collections import defaultdict
import regex as re
try:
    from .pretokenization_example import find_chunk_boundaries
except ImportError:
    # Allow direct script execution from this folder.
    from pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
BYTE_VOCAB_SIZE = 256

def bbpe(input_path:str, vocab_size:int, special_tokens:list[str])->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
     ## Usage
    assert vocab_size>=BYTE_VOCAB_SIZE+len(special_tokens), f"vocab contains at least {BYTE_VOCAB_SIZE} bytes and special tokens!"

    special_pat='|'.join(re.escape(t) for t in special_tokens)
    pre_token_cnts:defaultdict[str,int]=defaultdict(int)
    pre_token_splits:defaultdict[str,list[bytes,...]]=defaultdict(bytes)
    #pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")
            # Run pre-tokenization on your chunk and store the counts for each pre-token
            subchunks=re.split(special_pat,chunk)
            #print(subchunks)
            for subchunk in subchunks:
                for pre_token in re.finditer(PAT, subchunk):
                    pre_token_str=pre_token.group()
                    #print(repr(pre_token_str))
                    pre_token_splits[pre_token_str]=list(bytes([b]) for b in pre_token_str.encode('utf-8'))
                    pre_token_cnts[pre_token_str]+=1

    #initialize vocab
    vocab:dict[int,bytes]={b:bytes([b]) for b in range(BYTE_VOCAB_SIZE)}
    for i,t in enumerate(special_tokens):
        vocab[i+BYTE_VOCAB_SIZE]=t.encode('utf-8')
    merges=[]
    #merges up to vocab_size:
    while len(vocab)<vocab_size:
        #compute pair frequency
        pair_freqs=defaultdict(int)
        for pre_token,split in pre_token_splits.items():
            for i in range(len(split)-1):
                pair=(split[i],split[i+1])
                pair_freqs[pair]+=pre_token_cnts[pre_token]
        
        if len(pair_freqs)==0:
            print('not enough pairs before reach vocab size')
            break

        #get the max freqs
        best_pair,max_freq=max(pair_freqs.items(),key=lambda x: (x[1],x[0]))

        #merge best pair
        merged_byte=b''.join(best_pair)
        for pre_token, split in pre_token_splits.items():
            i=0
            # length of split changes, must use while
            while i<len(split)-1:
                if split[i:i+2]==list(best_pair):
                    split=split[:i]+[merged_byte]+split[i+2:]
                else:
                    i+=1
            pre_token_splits[pre_token]=split
        
        #collect merge rules and new token
        merges.append(best_pair)
        vocab[len(vocab)]=merged_byte

    return (vocab,merges)



if __name__=="__main__":
    vocab, merge=bbpe('./data/TinyStoriesV2-GPT4-valid.txt', 1000, ["<|endoftext|>"])
    print(vocab)
    print(merge)

                 

