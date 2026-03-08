from collections import defaultdict
import concurrent.futures
import regex as re
from pretokenization_example import find_chunk_boundaries
from line_profiler import profile

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
BYTE_VOCAB_SIZE = 256

def pre_tokenize_worker(args)->dict[str,int]:
    start, end, input_path, special_pat, PAT = args
    
    pre_token_local_cnts:defaultdict[str,int]=defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    # Run pre-tokenization on your chunk and store the counts for each pre-token
    subchunks=special_pat.split(chunk)
    #print(subchunks)
    for subchunk in subchunks:
        for pre_token in PAT.finditer(subchunk):
            pre_token_str=pre_token.group()
            #print(repr(pre_token_str))
            pre_token_local_cnts[pre_token_str]+=1
    return pre_token_local_cnts

@profile
def bbpe(input_path:str, vocab_size:int, special_tokens:list[str])->tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
     ## Usage
    assert vocab_size>=BYTE_VOCAB_SIZE+len(special_tokens), f"vocab contains at least {BYTE_VOCAB_SIZE} bytes and special tokens!"

    special_pat=re.compile('|'.join(re.escape(t) for t in special_tokens))
    pre_token_cnts:defaultdict[str,int]=defaultdict(int)
    pre_token_splits:defaultdict[str,list[bytes,...]]=defaultdict(bytes)
    #pre-tokenization
    with open(input_path, "rb") as f:
        num_processes = 1
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        process_args=[]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            process_args.append((start, end, input_path, special_pat, PAT))
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results=executor.map(pre_tokenize_worker,process_args)
            for pre_token_local_cnts in results:
                for pre_token_str, cnt in pre_token_local_cnts.items():
                    pre_token_cnts[pre_token_str]+=cnt
    
        for pre_token_str in pre_token_cnts:
            pre_token_splits[pre_token_str]=list(bytes([b]) for b in pre_token_str.encode('utf-8'))


    #initialize vocab
    vocab:dict[int,bytes]={b:bytes([b]) for b in range(BYTE_VOCAB_SIZE)}
    for i,t in enumerate(special_tokens):
        vocab[i+BYTE_VOCAB_SIZE]=t.encode('utf-8')
    merges=[]


    #compute pair frequency
    pair_freqs=defaultdict(int)
    pair_pre_token=defaultdict(set)
    for pre_token,split in pre_token_splits.items():
        for i in range(len(split)-1):
            pair=(split[i],split[i+1])
            pair_freqs[pair]+=pre_token_cnts[pre_token]
            #cache
            pair_pre_token[pair].add(pre_token)
    #merges up to vocab_size:
    while len(vocab)<vocab_size:

        if len(pair_freqs)==0:
            print('not enough pairs before reach vocab size')
            break

        #get the max freqs
        best_pair,max_freq=max(pair_freqs.items(),key=lambda x: (x[1],x[0]))

        #merge best pair
        merged_byte=b''.join(best_pair)
        for pre_token in pair_pre_token[best_pair]:
            split=pre_token_splits[pre_token]
            i=0
            # length of split changes, must use while
            while i<len(split)-1:
                if split[i:i+2]==list(best_pair):
                    if i>0:
                        pair_freqs[(split[i-1],split[i])]-=pre_token_cnts[pre_token]
                        if pair_freqs[(split[i-1],split[i])]==0:
                            del pair_pre_token[(split[i-1],split[i])]
                    if i<len(split)-2:
                        pair_freqs[(split[i+1],split[i+2])]-=pre_token_cnts[pre_token]
                        if pair_freqs[(split[i+1],split[i+2])]==0:
                            del pair_pre_token[(split[i+1],split[i+2])]
                    pair_freqs[best_pair]-=pre_token_cnts[pre_token]

                    split=split[:i]+[merged_byte]+split[i+2:]
                    if  i>0:
                        pair_freqs[(split[i-1],split[i])]+=pre_token_cnts[pre_token]
                        pair_pre_token[(split[i-1],split[i])].add(pre_token)
                    if i<len(split)-1:
                        pair_freqs[(split[i],split[i+1])]+=pre_token_cnts[pre_token]
                        pair_pre_token[(split[i],split[i+1])].add(pre_token)
                else:
                    i+=1
            pre_token_splits[pre_token]=split
        assert pair_freqs[best_pair]==0, f"pair freqs is not 0: {pair_freqs[best_pair]}"
        del pair_freqs[best_pair]
        
        #collect merge rules and new token
        merges.append(best_pair)
        vocab[len(vocab)]=merged_byte

    return (vocab,merges)



if __name__=="__main__":
    vocab, merge=bbpe('./cs336_basics/data/TinyStoriesV2-GPT4-valid.txt', 500, ["<|endoftext|>"])
    # print(vocab)
    # print(merge)

                 

