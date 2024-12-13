from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
import torch
import numpy as np 
from datasets import load_from_disk, Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from torch.nn.functional import softmax
import accelerate
from accelerate import Accelerator
import argparse
from data_converter import convert_dataset,convert_cnn_dataset
import argparse
# from GreedySTree_dy import GreedySTree
from transformers import AutoProcessor
from GreedySTree_dy_fast import GreedySTree
from Llama import LlamaForCausalLM_Attn
import time
from time import sleep
from utils import get_sampling_logits, _make_causal_mask, get_residual, cuda_graph_for_residual, cuda_graph_for_sampling_without_replacement, cuda_graph_for_sampling_with_replacement,cuda_graph_for_sampling_argmax 
import json
from Engine import GraphInferenceEngine, GraphInferenceEngineTG
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='model')
parser.add_argument('--target', type=str, help='target model')
parser.add_argument('--dataset', type=str, default="dataset/c4_small.json", help='dataset path')
parser.add_argument('--growmap', type=str, default="growmaps/68m_7b-64.pt", help='dataset path')
parser.add_argument('--start', type=int, default=0, help='start')
parser.add_argument('--end', type=int, default=200, help='end')
parser.add_argument('--T', type=float, default=0.6, help='temperature')
parser.add_argument('--P', type=float, default=0.9, help='top_p')
parser.add_argument('--DP', type=float, default=1.1, help='draft_top_p')
parser.add_argument('--D', type=int, default=1, help='depth')
parser.add_argument('--B', type=int, default=16, help='budget')
parser.add_argument('--W', type=int, default=16, help='max width')
parser.add_argument('--M', type=int, default=256, help='max length')
parser.add_argument('--g', type=int, default=1, help='gpu_number')
parser.add_argument('--Mode', type=str, default="greedy", help='tree mode')
parser.add_argument('--decay', type=float, default=0.85, help='decay')
parser.add_argument('--negative', action='store_true')
parser.add_argument('--static', action='store_true')
parser.add_argument('--offloading', action='store_true')
args = parser.parse_args()
print(args)



def simulation_greedy_with_tree_fast(target_model : GraphInferenceEngineTG, draft_model: GraphInferenceEngine, dataloader: DataLoader, T=0.6, top_p=0.9, 
            draft_top_p=1.1, budget=32, w=4, decay=0.85, negative=False, static=False, 
            max_length=512, residual_graph=None, grow_map=None, sampling_callables = None,
            sample_gather_indices = None, gpu=1):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    num_large_model_steps = 0
    total_time = 0.0
    dtype = torch.float16
    attn_mask = torch.full((max_length, max_length), torch.finfo(dtype).min, dtype=dtype, device='cuda:0')
    sequence = torch.tensor(list(range(max_length)), device='cuda:0').long().unsqueeze(-1)
    new_tokens_buffer =  torch.zeros(max_length).long().to('cuda:0')
    parents_buffer =  torch.zeros(max_length).long().to('cuda:0')
    position_ids = torch.zeros(max_length).long().to('cuda:0')
    with open('top_index_probabilities.txt', 'r') as file:
        #    self.indices = [int(line.strip()) for line in file.readlines()]
            probs = [float(line.strip()) for line in file.readlines()]
    probs = torch.tensor(probs, device='cuda:0')
    probs = probs +0.5
    probs = -(5/probs)
    
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            draft_kv_len = 0
            target_kv_len = 0
            attn_mask.fill_(torch.finfo(dtype).min)
            spectree = GreedySTree(prefix=input_ids.squeeze(0), device='cuda:0', temperature=T,
                                    top_p=top_p,
                                    draft_kv_len=draft_kv_len, target_kv_len=target_kv_len,
                                    draft_model_engine=draft_model, target_model_engine=target_model, max_length=max_length,probs = probs,grow_map=grow_map,
                                    attn_mask = attn_mask, sequence = sequence, new_tokens_buffer = new_tokens_buffer, 
                                    parents_buffer = parents_buffer, 
                                    position_ids = position_ids,
                                    residual_graph = residual_graph,
                                    sampling_callables=sampling_callables,
                                    sample_gather_indices = sample_gather_indices)
            torch.cuda.synchronize()
            t1 = time.time()
            while input_ids.shape[1] < 256 and terminate == False:
                spectree.construct_dynamic_tree()
                valid_tokens, draft_kv_len, target_kv_len, terminate = spectree.verify()
                num_decoding_steps += (valid_tokens.shape[0] - input_ids.shape[1])
                num_large_model_steps += 1
                input_ids = valid_tokens.unsqueeze(0)
                if (input_ids[0][-1] == 2) or (input_ids[0][-1] == 0): terminate = True
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            draft_model.clear_kv()
            target_model.clear_kv()
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}, large model step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps, num_large_model_steps))
    return num_decoding_steps / num_large_model_steps



def simulation_baseline(target_model : GraphInferenceEngineTG, dataloader: DataLoader, T=0.6, top_p=0.9, max_length=256):
    num_eval_steps = len(dataloader)
    num_decoding_steps = 0
    total_time = 0.0
    with torch.no_grad():
        for step, batch in tqdm(enumerate(dataloader), total=num_eval_steps):
            input_ids = batch['input_ids'][..., :128]
            labels = batch['labels'][..., :128]
            terminate = False
            if labels[0][-1] == -100: terminate = True
            position_ids = torch.arange(max_length).to('cuda:0').unsqueeze(0)
            storage_ids = torch.arange(max_length).to('cuda:0')
            attn_mask = _make_causal_mask((max_length, max_length), target_model.dtype, target_model.device)
            torch.cuda.synchronize()
            t1 = time.time()
            inner_decoding_step = 0
            start_length = 0
            while inner_decoding_step < 128 and terminate == False:
                if inner_decoding_step == 0:
                    start_length = input_ids.shape[1]
                    logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[:start_length],
                                                    position_ids = position_ids[..., :start_length], 
                                                    attn_mask=attn_mask[:start_length, :start_length][None, None, :, :])[0][-1]
                    
                else:
                    logits = target_model.inference(input_ids = input_ids, storage_ids=storage_ids[start_length + inner_decoding_step-1 : start_length + inner_decoding_step],
                                                    position_ids = position_ids[..., start_length + inner_decoding_step-1 : start_length + inner_decoding_step], 
                                                    attn_mask=attn_mask[start_length + inner_decoding_step-1 : start_length + inner_decoding_step, :start_length + inner_decoding_step][None, None, :, :])[0][-1]
                
                logits = get_sampling_logits(logits=logits, top_p=top_p, T=T)
                
                p = softmax(logits / T, dim=-1)
                new_token = p.multinomial(num_samples=1).unsqueeze(0)
                input_ids = new_token
                num_decoding_steps += 1
                inner_decoding_step += 1
                if input_ids[0][-1] == 2: 
                    terminate = True
            torch.cuda.synchronize()
            t2 = time.time()
            total_time += (t2 - t1)
            target_model.clear_kv()
            
    print("total time :{:.5f}s, latency :{:.5f}s, decoding step: {}".format(total_time, total_time / num_decoding_steps, num_decoding_steps))
    return num_decoding_steps



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.add_special_tokens({'pad_token': '[PAD]'})
if args.dataset == 'cnn':
    tokenized_dataset_eval = convert_cnn_dataset(tokenizer=tokenizer, seq_len = 256).select(list(range(args.start, args.end)))
elif args.dataset == 'openwebtext':
    tokenized_dataset_eval = load_from_disk("dataset/openwebtext_eval").select(list(range(args.start, args.end)))
else:
    tokenized_dataset_eval = convert_dataset(tokenizer=tokenizer,file_path=args.dataset).select(list(range(args.start, args.end)))

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
dataloader = DataLoader(tokenized_dataset_eval, batch_size=1, collate_fn=data_collator, shuffle=False)


if args.Mode == 'baseline':
    target_model =  GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0", offloading=args.offloading)
else:
    torch.distributed.init_process_group(backend='nccl')

    # Define rank and world size
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    # Assign GPU to the current process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)


    draft_model = GraphInferenceEngine(max_length=args.M, model_name_or_path = args.model, dtype = torch.float16, device="cuda:0")
    target_model = GraphInferenceEngineTG(max_length=args.M, model_name_or_path = args.target, dtype = torch.float16, device="cuda:0", offloading=args.offloading)
    for name, module in target_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            setattr(model, name, tensor_parallelize_linear_layer(module))
        
    for name, module in target_model.named_modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            setattr(model, name, tensor_parallel_attention(module))
    graph_capture_list = list(range(1, 129))
    draft_model.initialize_cuda_graph(graph_capture_list)
    residual_graph = cuda_graph_for_residual()
    path = args.growmap
    grow_map = torch.load(path)

    tree_size = grow_map["size"]
    idx_lists = grow_map["roots"]
    branch_lists = grow_map['branches']
    draft_step = len(grow_map["roots"])
    sampling_callables = {}
    sample_gather_indices = {}
    for i in range(draft_step - 1):
        idx_len = len(idx_lists[i])
        num_samples = max(branch_lists[i])
        sampling_callables[i] = cuda_graph_for_sampling_argmax(
            max_length=args.M, idx_len=idx_len, num_samples=num_samples,
            temperature=args.T, tree_size=tree_size)  
    for i in range(draft_step - 1):
        ith_gather_list = []
        max_num_samples = max(branch_lists[i])
        for j, branch in enumerate(branch_lists[i]):
            branch_index = torch.arange(branch, device="cuda:0", dtype=torch.long)
            branch_index = branch_index + j * max_num_samples
            ith_gather_list.append(branch_index)
        ith_gather_list = torch.cat(ith_gather_list)
        sample_gather_indices[i] = ith_gather_list
    

    

def tensor_parallelize_linear_layer(layer):
    in_features, out_features = layer.weight.shape
    local_out_features = out_features // world_size
    
    # Shard weights and biases across GPUs
    layer.weight.data = layer.weight.data[:, rank * local_out_features:(rank + 1) * local_out_features].contiguous()
    if layer.bias is not None:
        layer.bias.data = layer.bias.data[rank * local_out_features:(rank + 1) * local_out_features].contiguous()
    
    return layer

# Modify LLaMA model for tensor parallelism
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        setattr(model, name, tensor_parallelize_linear_layer(module))

# Define tensor-parallel forward pass for LLaMA's attention mechanism
def tensor_parallel_attention(attention_layer, hidden_states):
    """
    Forward pass for attention with tensor parallelism.
    """
    # Split the query, key, and value projections across GPUs
    query = attention_layer.q_proj(hidden_states)
    key = attention_layer.k_proj(hidden_states)
    value = attention_layer.v_proj(hidden_states)
    
    # Compute attention locally
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
    local_attention_output = torch.matmul(attention_scores, value)
    
    # Gather outputs across GPUs
    output_list = [torch.zeros_like(local_attention_output) for _ in range(world_size)]
    torch.distributed.all_gather(output_list, local_attention_output)
    
    # Concatenate outputs to form final result
    return torch.cat(output_list, dim=-1)




accelerator = Accelerator()
dataloader = accelerator.prepare(dataloader)

#warm up functions:

if args.Mode == 'baseline':
    simulation_baseline(target_model=target_model, dataloader=dataloader, T=args.T, top_p=args.P)
elif args.Mode == 'greedy':
    simulation_greedy_with_tree_fast(target_model=target_model, draft_model=draft_model, dataloader=dataloader, T=args.T, top_p=args.P, budget=args.B, draft_top_p=args.DP, w=args.W, negative=args.negative, decay=args.decay, static=args.static, 
                                     max_length=args.M, residual_graph = residual_graph, grow_map = grow_map, sampling_callables=sampling_callables, sample_gather_indices = sample_gather_indices, gpu = args.g)