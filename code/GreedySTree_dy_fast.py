import torch
from transformers import AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollatorForLanguageModeling, OPTForCausalLM, AutoTokenizer
from Llama import LlamaModel_Attn, LlamaForCausalLM_Attn
from torch.nn.functional import softmax
from copy import deepcopy
from tqdm import tqdm
from Tree import Tree
import time
import deepspeed
import gc
import tensor_parallel_linear_cuda
import generate_mask_cuda
from Engine import GraphInferenceEngine, GraphInferenceEngineTG
from torch.profiler import profile, record_function, ProfilerActivity
from utils import get_sampling_logits, make_tree_attention_mask, select_kv, ChildrenAccept, get_residual, cat_kv, _make_causal_mask
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch

class TreeUpdater:
    def __init__(self, device_ids=None):
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))
        self.world_size = len(self.device_ids)
        self.rank = 0  # For a single process per GPU setup, this can be dynamic in a multi-node setting
        self.local_device = torch.device(f'cuda:{self.rank}')

        # Initialize distributed process group
        dist.init_process_group(backend='nccl', init_method='env://', rank=self.rank, world_size=self.world_size)

    def update_tree_with_logits(self, logits, parent_indices, depths, values, cumulative_logits, new_logits, grow_step, updated_tensor):
        """Update the tree dynamically with new tokens and logits using multiple GPUs."""

        # Scatter parent nodes to each GPU
        parent_nodes = self.parent_nodes if grow_step != 0 else self.parent_nodes_0
        parent_nodes = parent_nodes[~updated_tensor[parent_nodes]]
        scattered_parent_nodes = torch.chunk(parent_nodes, self.world_size)

        # Scatter new_logits and perform `topk` on each GPU
        scattered_logits = torch.chunk(new_logits, self.world_size)
        scattered_updated_tensors = torch.chunk(updated_tensor, self.world_size)
        results = []

        for i, device_id in enumerate(self.device_ids):
            device = torch.device(f'cuda:{device_id}')
            parent_nodes_chunk = scattered_parent_nodes[i].to(device)
            logits_chunk = scattered_logits[i].to(device)
            updated_tensor_chunk = scattered_updated_tensors[i].to(device)

            if grow_step != 0:
                tokens_values, tokens_set = logits_chunk.topk(k=31)
            else:
                tokens_values, tokens_set = logits_chunk.topk(k=63)

            valid_mask = ~updated_tensor_chunk[parent_nodes_chunk]
            valid_parent_nodes = parent_nodes_chunk[valid_mask]

            valid_tokens_values = tokens_values[valid_parent_nodes - 1] if grow_step != 0 else tokens_values
            valid_tokens_set = tokens_set[valid_parent_nodes - 1] if grow_step != 0 else tokens_set

            results.append((valid_tokens_values, valid_tokens_set, valid_parent_nodes))

        # Gather results from all GPUs
        gathered_results = [torch.cat(tensors, dim=0) for tensors in zip(*results)]

        flattened_new_logits = gathered_results[0].flatten()
        flattened_new_tokens_set = gathered_results[1].flatten()
        replicated_parents = gathered_results[2].repeat_interleave(flattened_new_tokens_set.size(1))

        # Compute new depths and concatenate with existing tensors
        new_depths = depths[replicated_parents] + 1
        updated_logits = torch.cat([logits, flattened_new_logits])
        updated_parent_indices = torch.cat([parent_indices, replicated_parents])
        updated_depths = torch.cat([depths, new_depths])
        updated_values = torch.cat([values, flattened_new_tokens_set])

        # Update cumulative logits
        new_cumulative_logits = cumulative_logits[replicated_parents] + flattened_new_logits
        updated_cumulative_logits = torch.cat([cumulative_logits, new_cumulative_logits])

        # Prune tree if necessary
        if grow_step != 0:
            pruned_results = self.prune_tree(
                updated_logits, updated_parent_indices, updated_depths,
                updated_values, updated_cumulative_logits, keep=64, updated_tensor=updated_tensor
            )
            return pruned_results

        return updated_logits, updated_parent_indices, updated_depths, updated_values, updated_cumulative_logits, updated_tenso


class GreedySTree(Tree):
    def __init__(self, 
                 #draft_model :LlamaForCausalLM_Attn, 
                 draft_model_engine :GraphInferenceEngine,
                 target_model_engine :GraphInferenceEngineTG,
                 prefix :torch.LongTensor,
                 temperature :float = 0.6,
                 top_p: float = 0.9,
                 draft_kv_len = 0,
                 target_kv_len = 0,
                 max_length = 256,
                 device :str = 'cpu',
                 max_target_seq = 256,
                 vocab_size = 32000,
                 probs = None,
                 grow_map = None,
                 attn_mask = None, 
                 sequence = None, 
                 new_tokens_buffer = None, 
                 parents_buffer = None, 
                 position_ids = None,
                 residual_graph = None,
                 sampling_callables = None,
                 sample_gather_indices = None,
                 gpu = None) -> None:
        super().__init__(device=device, max_length=max_length)
        assert self.max_length == draft_model_engine.engine.max_length
        self.max_target_seq = max_target_seq
        #self.draft_model = draft_model.to(self.device).eval()
        # p 0.9 t 0.01
        self.draft_model_engine = draft_model_engine
        self.target_model_engine = target_model_engine
        self.temperature = temperature
        self.top_p = top_p
        self.residual_graph = residual_graph
        self.grow_map = grow_map
        self.sampling_callables = sampling_callables
        self.sample_gather_indices = sample_gather_indices
        self.draft_step = 7
        self.grow_map_roots_gpu = []
        for x in self.grow_map["roots"]:
             self.grow_map_roots_gpu.append(torch.Tensor(x).to(self.device).long())
        self.Successors = self.grow_map["Successors"]
        tree_mask :torch.Tensor = self.grow_map["mask"].to(self.device)
        tree_mask = (tree_mask == 0).type(self.dtype)
        
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        self.initialize(attn_mask, sequence, new_tokens_buffer, parents_buffer, position_ids, None,None)
        self.set_prefix(prefix=prefix)
        self.tree_size = self.grow_map["size"]
        # print(self.tree_size)
        self.tree_mask = tree_mask
        self.root = TreeNode(gpu)
        self.full_attn_mask[self.max_length - self.tree_size + 1: self.max_length, self.max_length - self.tree_size + 1: self.max_length] = tree_mask[1:, 1:]
        self.tree = TreeUpdater()
        total_nodes = len(prefix) + self.tree_size - 1
        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes, self.max_length - total_nodes: 2 * self.max_length - total_nodes]
        self.ground_truth_len = len(prefix)
        self.r = torch.rand(len(position_ids), dtype=self.dtype).to(self.device)
        
        self.position_ids[len(prefix) : len(prefix) + self.tree_size - 1] = (self.grow_map["depth"][1:].to(self.device) + len(prefix) - 1)
        self.storage_ids = torch.arange(self.max_length).to(self.device)
        self.depth = self.grow_map["depth"][1:].to(self.device)
        self.logits = torch.tensor([0.0], dtype=torch.float).to(self.device)
        self.parent_indices = torch.tensor([-1], dtype=torch.long).to(self.device)
        self.depths = torch.tensor([0], dtype=torch.long).to(self.device)
        self.values = torch.tensor([0], dtype=torch.long).to(self.device)
        self.cumulative_logits = torch.tensor([0.0], dtype=torch.float).to(self.device)
        self.draft_logits = torch.zeros((self.max_length, vocab_size), dtype=self.dtype).to(self.device)
        if draft_kv_len == 0:
            draft_model_outputs = self.draft_model_engine.inference(input_ids=self.tokens[:self.num_nodes].unsqueeze(0), 
                                storage_ids=self.storage_ids[:self.num_nodes], 
                                position_ids=self.position_ids[:self.num_nodes].unsqueeze(0),
                                attn_mask=self.attn_mask[:self.num_nodes][None, None, :, :])
            self.draft_logits[0] :torch.FloatTensor= draft_model_outputs[...,-1,:][0]
        
        else:
            draft_model_outputs = self.draft_model_engine.inference(input_ids = self.tokens[draft_kv_len: self.num_nodes].unsqueeze(0), 
                                                    storage_ids=self.storage_ids[draft_kv_len: self.num_nodes],
                                                    position_ids=self.position_ids[draft_kv_len: self.num_nodes].unsqueeze(0),
                                                    attn_mask=self.attn_mask[draft_kv_len: self.num_nodes][None, None, :, :])
            self.draft_logits[0] :torch.FloatTensor = draft_model_outputs[...,-1,:][0]
        self.draft_kv_len = self.num_nodes
        
        self.target_kv_len = target_kv_len
        self.mask = torch.eye(63, dtype=torch.bool).to(self.device)
        self.seq_to_use = list(range(self.max_length))
        self.attn_engine = AttentionMaskEngine(device=self.device, dtype=self.dtype)
        self.parent_nodes = torch.arange(1, 64, device=self.device)
        self.parent_nodes_0 = torch.arange(1, device=self.device)
        self.parent_to_children = self.generate_parent_to_children_map()
        self.probs = probs
    def generate_parent_to_children_map(self):
        parent_to_children = {}
        for child_idx, parent_idx in enumerate(self.parent_indices):
            if parent_idx != -1:  # Skip the root node
                if parent_idx not in parent_to_children:
                    parent_to_children[parent_idx] = []
                parent_to_children[parent_idx].append(child_idx)
        return parent_to_children

    def generate_attention_mask(self, parent_indices):
        parent_indices = parent_indices[1:] - 1
        num_nodes = len(parent_indices)

        # Initialize the mask on GPU
        mask = torch.eye(num_nodes, device=self.device, dtype=self.dtype)
        
        # Track which indices need to be updated
        to_update = torch.arange(num_nodes, device=self.device)

        while to_update.numel() > 0:
            parent_index = parent_indices[to_update]
            valid = parent_index != -1
            if valid.any():
                valid_indices = to_update[valid]
                parent_valid_indices = parent_index[valid]
                
                # Use advanced indexing for efficient updates
                mask[valid_indices, parent_valid_indices] = 1
                mask[valid_indices] = torch.max(mask[valid_indices], mask[parent_valid_indices])
                
                # Update to_update and parent_indices
                to_update = valid_indices
                parent_indices[to_update] = parent_indices[parent_valid_indices]
            else:
                break

        return mask

    def prune_tree(self,logits, parent_indices, depths, values, cumulative_logits, keep=63,updated_tensor=0):
        """Prune the tree to keep up to 'keep' nodes based on unique cumulative logits, prioritizing uniqueness."""
        # Sort nodes based on cumulative logits to prioritize higher values
        # t1 = time.time()
        sorted_indices = torch.argsort(cumulative_logits, descending=True)[:keep]
        keep_nodes = torch.zeros_like(logits, dtype=torch.bool)
        keep_nodes[sorted_indices] = True
        # Create a mapping from old indices to new indices
        old_to_new = torch.full_like(logits, -1, dtype=torch.long)
        old_to_new[keep_nodes] = torch.arange(len(sorted_indices), dtype=torch.long).to(self.device)
        # keep_nodes[0] = True  # Ensure the root is kept
        pruned_logits = logits[keep_nodes]
        pruned_parent_indices = old_to_new[parent_indices[keep_nodes]]
        # pruned_parent_indices = parent_indices[keep_nodes]
        # pruned_parent_indices = parent_indices[keep_nodes]
        pruned_depths = depths[keep_nodes]
        pruned_values = values[keep_nodes]
        updated_tensor = updated_tensor[keep_nodes]
        pruned_cumulative_logits = cumulative_logits[keep_nodes]


        return pruned_logits, pruned_parent_indices, pruned_depths, pruned_values, pruned_cumulative_logits,updated_tensor
    def update_tree_with_logits(self,logits, parent_indices, depths, values, cumulative_logits, new_logits, grow_step, updated_tensor):
        """Update the tree dynamically with new tokens and logits."""
        # t1 = time.time()
        if grow_step != 0:
            # new_logits = new_logits + self.probs
            new_tokens_values, new_tokens_set = new_logits.topk(k=31)
            # new_tokens_values += self.probs[new_tokens_set]
        else:
            # new_logits = new_logits + self.probs
            new_tokens_values, new_tokens_set = new_logits.topk(k=63)
            # new_tokens_values += self.probs[new_tokens_set]
        if grow_step != 0:
            mask = ~updated_tensor[self.parent_nodes]
            parent_nodes = self.parent_nodes[mask]
        else: 
            mask = ~updated_tensor[self.parent_nodes_0]
            parent_nodes = self.parent_nodes_0[mask]
        # t6 = time.time()
        # time_taken2 = t6 - t5
        # print(f"Time taken to create new parent nodes: {time_taken2:.6f} seconds")
        # Filter the new logits and token sets to only consider valid parent nodes
        # t7 = time.time()
        if grow_step!=0:
            valid_new_tokens_values = new_tokens_values[parent_nodes-1]
            valid_new_tokens_set = new_tokens_set[parent_nodes-1]
        else:
            valid_new_tokens_values = new_tokens_values
            valid_new_tokens_set = new_tokens_set
        # t8 = time.time()
        # time_taken3 = t8 - t7
        # print(f"Time taken to fill token set: {time_taken3:.6f} seconds")
        # t9 = time.time()
        num_new_tokens = valid_new_tokens_set.size(1)
        replicated_parents = parent_nodes.repeat_interleave(num_new_tokens)
        # t10 = time.time()
        # time_taken4 = t10 - t9
        # print(f"Time taken to develop parent node tree: {time_taken4:.6f} seconds")
        # Flatten the filtered new logits and corresponding token indices
        # t11 = time.time()
        flattened_new_logits = valid_new_tokens_values.flatten()
        flattened_new_tokens_set = valid_new_tokens_set.flatten()
        # t12 = time.time()
        # time_taken5 = t12 - t11
        # print(f"Time taken to flatten: {time_taken5:.6f} seconds")
        # Compute the new depths for the new nodes
        # t13 = time.time()
        new_depths = depths[replicated_parents] + 1
        # t14 = time.time()
        # time_taken6 = t14 - t13
        # print(f"Time taken to update depth: {time_taken6:.6f} seconds")
        # Concatenate the old and new tensors
        # t15 = time.time()
        updated_logits = torch.cat([logits, flattened_new_logits]) #pre allocate
        updated_parent_indices = torch.cat([parent_indices, replicated_parents])
        updated_depths = torch.cat([depths, new_depths])
        updated_values = torch.cat([values, flattened_new_tokens_set])
        updated_tensor = torch.zeros_like(updated_logits, dtype=torch.bool)
        # Mark the parent nodes as expanded
        updated_tensor[:63] = True
        # t16 =time.time()
        # time_taken7 = t16 - t15
        # print(f"Time taken to update tensor: {time_taken7:.6f} seconds")

        # Update cumulative logits
        # t18 = time.time()
        new_cumulative_logits = cumulative_logits[replicated_parents] +  flattened_new_logits
        updated_cumulative_logits = torch.cat([cumulative_logits, new_cumulative_logits])
        # t19 = time.time()
        # time_taken8 = t19 - t18
        # print(f"Time taken to update cumulative logits: {time_taken8:.6f} seconds")
        if grow_step == 0:
            return updated_logits, updated_parent_indices, updated_depths, updated_values, updated_cumulative_logits, 0
        # t20 = time.time()
        pruned_logits, pruned_parent_indices, pruned_depths, pruned_values, pruned_cumulative_logits,updated_tensor = self.prune_tree(
            updated_logits, updated_parent_indices, updated_depths, updated_values, updated_cumulative_logits, keep=64, updated_tensor = updated_tensor
        )
        # t21 = time.time()
        # time_taken9 = t21 - t20
        # print(f"Time taken to prune tree with logits: {time_taken9:.6f} seconds")
        return pruned_logits, pruned_parent_indices, pruned_depths, pruned_values, pruned_cumulative_logits, updated_tensor
    
    @torch.inference_mode()
    def collective_grow_dynamic(self, benchmark=False, grow_step = None):
        
        if benchmark:
            x1 = 0.0
            x2 = 0.0
        
        if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
        if grow_step == 0:
           sampling_logits = self.draft_logits[0].unsqueeze(0)
           updated_tensor = torch.zeros_like(self.logits, dtype=torch.bool).to(self.device)
        elif grow_step == 1:
           sampling_logits = self.draft_logits[1:64]
           self.updated_tensor = torch.zeros_like(self.logits, dtype=torch.bool).to(self.device)
           self.updated_tensor[0] = 1
        else:
           sampling_logits = self.draft_logits[1:64]
        torch.cuda.synchronize()
        t1 = time.time()
        if grow_step == 0:
            self.logits, self.parent_indices, self.depths, self.values, self.cumulative_logits, updated_tensor = self.tree.update_tree_with_logits(
            self.logits, self.parent_indices, self.depths, self.values, self.cumulative_logits, sampling_logits, grow_step,updated_tensor=updated_tensor
            )
        else:
            self.logits, self.parent_indices, self.depths, self.values, self.cumulative_logits, self.updated_tensor = self.tree.update_tree_with_logits(
            self.logits, self.parent_indices, self.depths, self.values, self.cumulative_logits, sampling_logits, grow_step,updated_tensor=self.updated_tensor
            )
        # nodes = [node for node in get_all_nodes(self.root)]
        # self.update_tree_with_logits_1(sampling_logits, nodes, grow_step)
        torch.cuda.synchronize()
        t2 = time.time()
        time_taken1 = t2 - t1
        # print(f"Time taken to update tree with logits: {time_taken1:.6f} seconds")
        # print(torch.tensor(get_all_nodes_value(self.root)[1:]).to(self.device)==self.values[1:])
        self.tokens[self.num_nodes: self.num_nodes + 63] = self.values[1:]
        # torch.tensor(get_all_nodes_value(self.root)[1:])
        if benchmark:
                    torch.cuda.synchronize()
                    t2 = time.time()
                    x1 += (t2 - t1)

        position_ids = self.depths[1:]+self.num_nodes-1

        attn_mask = self.attn_mask[self.num_nodes: self.num_nodes+63]
        torch.cuda.synchronize()
        t7 = time.time()
        if grow_step == 0:
            tree_mask = self.mask
            # tree_mask1 = self.mask
        else:
            tree_mask = generate_attention_mask_cuda(self.parent_indices)
        torch.cuda.synchronize()
        t10 = time.time()
        time_taken5 = t10 - t7
        # print(f"Time taken to generate attention mask1: {time_taken5:.6f} seconds")
        tree_mask = (tree_mask == 0).type(self.dtype)
        tree_mask.masked_fill_(tree_mask > 0, torch.finfo(self.dtype).min)
        torch.cuda.synchronize()
        t8 = time.time()
        time_taken4 = t8 - t7
        # print(f"Time taken to generate attention mask: {time_taken4:.6f} seconds")
        attn_mask[0:63 , self.num_nodes:self.num_nodes+63] = tree_mask
        attn_mask = attn_mask[None, None, :, :]
        t5 = time.time()
        draft_model_outputs = self.draft_model_engine.graph_inference(
            input_ids = self.tokens[self.num_nodes: self.num_nodes+63].unsqueeze(0),
            position_ids = position_ids.unsqueeze(0),
            attn_mask = attn_mask,
            storage_ids=self.storage_ids[self.num_nodes: self.num_nodes+63]
        )
        t6 = time.time()
        time_taken3 = t6 - t5
        # print(f"Time taken to forward draft model: {time_taken3:.6f} seconds")
        self.draft_kv_len = self.num_nodes+63
        self.draft_logits[1:64] = draft_model_outputs[0][-63:]
        if benchmark:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    x2 += (t3 - t2)
        if benchmark:
            return x1, x2
        return position_ids, tree_mask
 
 
    @torch.inference_mode()
    def accept_step(self, parent_id :int) ->ChildrenAccept:
        logits_id = parent_id - (self.ground_truth_len - 1)
        p = self.target_logits[logits_id]
        draft_logits = self.draft_logits[logits_id]
        
        children = self.Successors[logits_id]

        # target_token = p.argmax(dim=-1)
        if len(children) == 0:
            return (-1, p)
        
        for pos in children:

            token = self.tokens[pos + (self.ground_truth_len - 1)]
            q = softmax(draft_logits / self.temperature, dim=-1)
            r = self.r[pos + (self.ground_truth_len - 1)]
            
            # if token == target_token:
            
            #      return (pos + (self.ground_truth_len - 1), None)
            
            if p[token] >= r * q[token]:
                #return ChildrenAccept(accept_mark=0, token=token, position=pos + (self.ground_truth_len - 1), successor_order=idx)
                return (pos + (self.ground_truth_len - 1), None)
            else:
                p = self.residual_graph(p, q)
                draft_logits[token] = torch.finfo(self.dtype).min
        return (-1, p)
  
    @torch.inference_mode()
    def verify(self, benchmark = False):
        new_node_num = (self.num_nodes - self.ground_truth_len + 1)
        if self.target_kv_len == 0:
            start_pos = 0
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask[end_pos-63:end_pos,end_pos-63:end_pos] = self.draft_tree_mask
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            self.position_ids[end_pos-63:end_pos] = self.draft_postion_ids
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                    position_ids = self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask, 
                                    storage_ids=self.storage_ids[start_pos : end_pos])
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor= target_model_outputs[0][self.ground_truth_len - 1:]

        else:
            start_pos = self.target_kv_len
            end_pos = self.num_nodes
            attn_mask = self.attn_mask[start_pos: end_pos, :end_pos]
            attn_mask[1:,end_pos-63:end_pos] = self.draft_tree_mask
            attn_mask = attn_mask[None, None, :, :]
            if benchmark:
                torch.cuda.synchronize()
                t1 = time.time()
            self.position_ids[start_pos+1 : end_pos] = self.draft_postion_ids
            torch.cuda.synchronize()
            t1 = time.time()
            target_model_outputs = self.target_model_engine.inference(input_ids = self.tokens[start_pos : end_pos].unsqueeze(0), 
                                        position_ids =self.position_ids[start_pos : end_pos].unsqueeze(0), attn_mask = attn_mask,
                                        storage_ids=self.storage_ids[start_pos : end_pos])
            torch.cuda.synchronize()
            t2 = time.time()
            time_taken = t2-t1
            # print(f"Time taken to forward target model: {time_taken:.6f} seconds") 
            if benchmark:
                torch.cuda.synchronize()
                t2 = time.time()
            self.target_logits :torch.FloatTensor = target_model_outputs[0][-(new_node_num):]
        
        self.target_logits = get_sampling_logits(logits=self.target_logits, top_p=self.top_p, T=self.temperature, replicate=False)
        self.target_logits = softmax(self.target_logits / self.temperature, dim=-1)
        #self.target_token = self.target_logits.argmax(dim=-1)
        self.target_token = self.target_logits.multinomial(num_samples=1)
        accept_list = self.seq_to_use[:self.ground_truth_len]
        
        terminal = False
        while True:
            parent_id = accept_list[-1]
            pos, res = self.accept_step(parent_id=parent_id)
            if pos != -1:
                accept_list.append(pos)
                if self.tokens[pos] == 0 or self.tokens[pos] == 2:
                     terminal = True
                     break
            else:
                residual = res
                break
        if benchmark:
            torch.cuda.synchronize()
            t3 = time.time()
        accept_length = len(accept_list)
        if not terminal:
            if torch.isnan(residual).any():
                terminal = True
            else:
                self.tokens[accept_length] = residual.multinomial(num_samples=1, replacement=True)
                self.tokens[:accept_length] = self.tokens[accept_list]

        self.draft_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)
        self.target_model_engine.engine.kv_cache.gather_kv_incremental(accept_list[self.ground_truth_len:], self.ground_truth_len)

        if not terminal:
            if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
                return self.tokens[:accept_length+1], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
            self.prepare_for_next_iter(accept_list, self.tokens[:accept_length+1])
            return self.tokens[:accept_length+1], accept_length, accept_length, terminal
        else:
             if benchmark:
                torch.cuda.synchronize()
                t4 = time.time()
                return self.tokens[:accept_length], accept_length, accept_length, t2 - t1, t3-t2, t4 - t3, terminal
             return self.tokens[:accept_length], accept_length, accept_length, terminal
    def verbose(self):
        super().verbose()
    def construct_dynamic_tree(self, benchmark = False):
        if benchmark:
            sample_time = 0
            compute_time = 0
        for i in range(self.draft_step - 1):
                if benchmark:
                        _, t1, t2 = self.collective_grow_static(self.grow_map_roots_gpu[i], self.grow_map['branches'][i], benchmark=benchmark, grow_step=i)
                        sample_time += t1
                        compute_time += t2   
                else:
                        position_ids, tree_mask = self.collective_grow_dynamic(grow_step=i)
        self.draft_postion_ids = position_ids
        self.draft_tree_mask = tree_mask
        self.num_nodes = self.num_nodes+63
        torch.cuda.synchronize()
        t3 = time.time()
        self.Successors = generate_successors_list(self.parent_indices)
        torch.cuda.synchronize()
        t4 = time.time()
        time_taken = t4-t3
        # print(f"Time taken to generate successor: {time_taken:.6f} seconds") 
        # self.Successors1 = generate_successors_list1(self.root)
        
        if benchmark:
            return sample_time, compute_time
        else:
            return None
    def prepare_for_next_iter(self, accept_list: list[int], valid_tokens :torch.LongTensor):
        if len(accept_list) + 1 > self.max_target_seq:
              return 
        self.tokens[:len(valid_tokens)] = valid_tokens
        self.position_ids[:len(accept_list)] =  self.position_ids[accept_list]
        self.position_ids[len(accept_list)] = len(accept_list) 
        self.position_ids[len(valid_tokens) : len(valid_tokens) + self.tree_size - 1] = (self.depth + len(valid_tokens) - 1)
        self.ground_truth_len = len(valid_tokens)
        self.num_nodes = len(valid_tokens)

        total_nodes = len(valid_tokens) + self.tree_size - 1

        self.attn_mask = self.full_attn_mask[self.max_length - total_nodes: 2 * self.max_length - total_nodes, self.max_length - total_nodes: 2 * self.max_length - total_nodes]

        
        draft_model_outputs = self.draft_model_engine.graph_inference(input_ids = self.tokens[len(accept_list): self.num_nodes].unsqueeze(0), 
                                                    storage_ids=self.storage_ids[len(accept_list): self.num_nodes],
                                                    position_ids=self.position_ids[len(accept_list): self.num_nodes].unsqueeze(0),
                                                    attn_mask=self.attn_mask[len(accept_list): self.num_nodes][None, None, :, :])
        
        self.draft_logits[0] :torch.FloatTensor = draft_model_outputs[...,-1,:][0]
        self.logits = torch.tensor([0.0], dtype=torch.float).to(self.device)
        self.parent_indices = torch.tensor([-1], dtype=torch.long).to(self.device)
        self.depths = torch.tensor([0], dtype=torch.long).to(self.device)
        self.values = torch.tensor([0], dtype=torch.long).to(self.device)
        self.cumulative_logits = torch.tensor([0.0], dtype=torch.float).to(self.device)
        self.draft_kv_len = self.num_nodes
        self.target_kv_len = len(accept_list)


        


