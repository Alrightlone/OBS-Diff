import time 
import heapq 
import torch 
import torch.nn as nn 
from .OBS_Diff import OBS_Diff 
from .OBS_Diff_Structured import OBS_Diff_Structured, OBS_Diff_Structured_Joint_Attn
from collections import defaultdict
from .dataloader import get_loaders 
from collections import OrderedDict
import numpy as np
import torch_pruning as tp


def get_module_by_name(layer, name):
    module = layer
    for attr in name.split('.'):
        module = getattr(module, attr)
    return module

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(transformer_model, target_names):
    print("\n" + "="*50)
    print("Calculating Sparsity for Target Modules...")
    print("="*50)

    all_linear_layers = find_layers(transformer_model)

    individual_sparsities = OrderedDict()
    
    total_target_params = 0
    total_target_zeros = 0


    for layer_name, layer_module in all_linear_layers.items():
        is_target = False
        for target_suffix in target_names:
            if layer_name.endswith(target_suffix):
                is_target = True
                break
        
        if is_target:
            weight = layer_module.weight
            
            if weight.numel() > 0:
                zeros = torch.sum(weight == 0).item()
                total = weight.numel()
                sparsity_percentage = (zeros / total) * 100 if total > 0 else 0
                
                individual_sparsities[layer_name] = sparsity_percentage
                
                total_target_zeros += zeros
                total_target_params += total

    print("--- Individual Module Sparsity ---")
    if not individual_sparsities:
        print("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            print(f"{name:<50s} | Sparsity: {sparsity:.4f}%")
            
    print("--- Aggregate Sparsity for Targets ---")
    if total_target_params > 0:
        aggregate_percentage = (total_target_zeros / total_target_params) * 100
        print(f"Total Parameters in Target Modules : {total_target_params}")
        print(f"Zero Parameters in Target Modules  : {total_target_zeros}")
        print(f"Overall Sparsity of Target Modules : {aggregate_percentage:.4f}%")
    else:
        print("No parameters found in target modules.")
    
    print("\n" + "="*50)
    print("Calculation finished.")
    print("="*50)

    return aggregate_percentage

def check_size(transformer_model, target_names):
    print("\n" + "="*50)
    print("Checking Size for Target Modules...")
    print("="*50)

    all_linear_layers = find_layers(transformer_model)

    individual_sparsities = OrderedDict()
    individual_sparsities_structured = OrderedDict()
    individual_sparsities_bias = OrderedDict()
    total_target_params = 0
    total_target_zeros = 0

    if "ff.net.2" in target_names:
        target_names.append("ff.net.0.proj")
    if "ff_context.net.2" in target_names:
        target_names.append("ff_context.net.0.proj")
    if "attn.to_out.0" in target_names:
        target_names.append("attn.to_add_out")
        target_names.append("attn.to_q")
        target_names.append("attn.to_k")
        target_names.append("attn.to_v")
        target_names.append("attn.add_k_proj")
        target_names.append("attn.add_q_proj")
        target_names.append("attn.add_v_proj")
    for layer_name, layer_module in all_linear_layers.items():
        is_target = False
        for target_suffix in target_names:
            if layer_name.endswith(target_suffix):
                is_target = True
                break
        
        if is_target:
            weight = layer_module.weight
            
            if weight.numel() > 0:
                zeros = torch.sum(weight == 0).item()
                total = weight.numel()
                sparsity_percentage = (zeros / total) * 100 if total > 0 else 0

                individual_sparsities[layer_name] = sparsity_percentage
                individual_sparsities_structured[layer_name] = layer_module.weight.shape
                individual_sparsities_bias[layer_name] = layer_module.bias.shape
                total_target_zeros += zeros
                total_target_params += total

    print("\n--- Individual Module Sparsity ---")
    if not individual_sparsities:
        print("No target modules found.")
    else:
        for name, sparsity in individual_sparsities.items():
            print(f"{name:<50s} | {individual_sparsities_structured[name]} {individual_sparsities_bias[name]}")
            
    
    
    print("\n" + "="*50)
    print("Calculation finished.")
    print("="*50)


def group_modules_with_parallelism(target_pruned_modules, num_groups):

    parallel_sets_rules = [
        {"attn.to_q", "attn.to_k", "attn.to_v", "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj"},
        {"attn.to_out.0", "attn.to_add_out"},
        {"ff_context.net.0.proj", "ff.net.0.proj"},
        {"ff.net.2", "ff_context.net.2"},
    ]
    modules_by_block = defaultdict(list)
    for block_idx, name in target_pruned_modules:
        modules_by_block[block_idx].append(name)

    groupable_items = []
    
    for block_idx in sorted(modules_by_block.keys()):
        block_modules = set(modules_by_block[block_idx])
        processed_modules = set()

        for p_set in parallel_sets_rules:
            intersection = block_modules.intersection(p_set)
            if intersection:
                parallel_unit = [(block_idx, name) for name in sorted(list(intersection))] # 排序以保证确定性
                groupable_items.append(parallel_unit)
                processed_modules.update(intersection)
        
        remaining_modules = block_modules - processed_modules
        for name in sorted(list(remaining_modules)):
            groupable_items.append([(block_idx, name)])

    num_items = len(groupable_items)
    print(f"num_items: {num_items}")
    if num_items == 0:
        return []

    group_size = num_items // num_groups
    remainder = num_items % num_groups
    
    if group_size == 0:
        group_size = 1
        num_groups = num_items
        remainder = 0
    
    final_groups = []
    start_index = 0
    for i in range(num_groups):
        end_index = start_index + group_size + (1 if i < remainder else 0)
        
        # e.g., [ [(0, 'ff.net.2'), (0, 'ff_context.net.2')], [(0, 'attn.to_q')] ]
        current_chunk = groupable_items[start_index:end_index]
        
        flat_group = [module for unit in current_chunk for module in unit]
        final_groups.append(flat_group)
        
        start_index = end_index

    return final_groups

def create_hook_fn(block_idx, layer_name, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        pruner = pruner_dict[(block_idx, layer_name)]
        
        # get the input data
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples
        
        input_data = input_data * np.sqrt(current_weight)
        # call add_batch, pass the weighted input data
        pruner.add_batch(input_data, output.data, W_new)
        #msg = f"Updated Hessian for Block {block_idx}, {layer_name}, Step {step}, Input Shape: {input[0].shape}, Weight: {timestep_weight[step]:.4f}"
        #logger.info(msg)  
    return hook_fn

def create_hook_fn_Joint_Attn(block_idx, layer_name, pruner_dict, timestep_weight):
    def hook_fn(module, input, output):
        step = step_info["current"]
        if layer_name == "attn.to_add_out":
            pruner = pruner_dict[(block_idx, "attn.to_out.0")]
        else:
            pruner = pruner_dict[(block_idx, layer_name)]
        
        # get the input data
        input_data = input[0].data

        current_weight = timestep_weight[step]
        num_samples = input_data.shape[0]
        W_new = current_weight * num_samples
        
        input_data = input_data * np.sqrt(current_weight)
        # call add_batch, pass the weighted input data

        #print(f"input_data.shape: {input_data.shape}")
        #print(f"layer_name: {layer_name}")
        pruner.add_batch(input_data, output.data, layer_name, W_new)
        #print(f"input_data.shape: {input_data.shape}")
        #print(f"layer_name: {layer_name}")
        #msg = f"Updated Hessian for Block {block_idx}, {layer_name}, Step {step}, Input Shape: {input[0].shape}, Weight: {timestep_weight[step]:.4f}"
        #print(msg)  
    return hook_fn

step_info = {"current": 0}
# callback function, update the step value in step_info after each denoising step
def callback_on_step_end(pipeline, step, timestep, callback_kwargs):
    step_info["current"] += 1
    return callback_kwargs

@torch.no_grad()
def prune_OBS_Diff(args, pipe, target_modules,  dev, prune_n=0, prune_m=0, timestep_weight=None):
    print('Starting ...')
    dataloader = get_loaders(
        args.dataset,
        num_samples=args.num_samples
    )
    blocks = pipe.transformer.transformer_blocks
    target_pruned_modules = []


    for i in range(args.minlayer, args.maxlayer):
        block = blocks[i]
        all_module_dict = find_layers(block)
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    # Divide target_pruned_modules into num_pruned_groups groups
    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    print(f"\n intelligently divided {num_modules} modules into {len(modules_groups)} groups:")

    for g_idx, group in enumerate(modules_groups):
        print(f"Group {g_idx + 1}: {[(block_idx, name) for block_idx, name in group]}")
    for g_idx, group_modules in enumerate(modules_groups):
        print(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        # Initialize pruner and hooks for this group
        pruner_dict = {}
        hooks = []
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]
            pruner_dict[(block_idx, module_name)] = OBS_Diff(module, args)
            hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
            hooks.append(module.register_forward_hook(hook_fn))
     
        # Run prompts in dataloader to collect activations
        print(f"Running diffusion for group {g_idx + 1} to collect activations...")
        # consider batch_size
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
         
            print(f"  Prompts {i}: {prompts}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        # Remove hooks for this group
        for hook in hooks:
            hook.remove()
        
        # Execute pruning for this group
        print(f"Pruning group {g_idx + 1}...")
        for block_idx, module_name in group_modules:
            print(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            pruner_dict[(block_idx, module_name)].fasterprune(
                sparsity=sparsity,
                percdamp=args.percdamp,
                prunen=prune_n,
                prunem=prune_m
            )
            pruner_dict[(block_idx, module_name)].free()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        print(f"Group {g_idx + 1} pruning completed.")


@torch.no_grad()
def prune_OBS_Diff_Structured(args, pipe, target_modules, dev, timestep_weight=None):
    print('Starting ...')
    dataloader = get_loaders(
        args.dataset,
        num_samples=args.num_samples  
    )

    blocks = pipe.transformer.transformer_blocks
    target_pruned_modules = []


    for i in range(args.minlayer, args.maxlayer):
        block = blocks[i]
        all_module_dict = find_layers(block)
        print(f"all_module_dict: {all_module_dict}")
        for name in target_modules:
            if name in all_module_dict:
                target_pruned_modules.append((i, name))

    # Divide target_pruned_modules into num_pruned_groups groups
    modules_groups = group_modules_with_parallelism(target_pruned_modules, args.num_pruned_groups)

    num_modules = len(target_pruned_modules)
    print(f"\n intelligently divided {num_modules} modules into {len(modules_groups)} groups:")

    for g_idx, group in enumerate(modules_groups):
        print(f"Group {g_idx + 1}: {[(block_idx, name) for block_idx, name in group]}")


    # Process each group
    for g_idx, group_modules in enumerate(modules_groups):
        print(f"\nProcessing group {g_idx + 1}/{len(modules_groups)}...")
        
        # Initialize pruner and hooks for this group
        pruner_dict = {}
        hooks = []
        for block_idx, module_name in group_modules:
            block = blocks[block_idx]
            all_module_dict = find_layers(block)
            module = all_module_dict[module_name]
            if module_name == "ff.net.2" or module_name == "ff_context.net.2":
                pruner_dict[(block_idx, module_name)] = OBS_Diff_Structured(module, block_idx, args)

                hook_fn = create_hook_fn(block_idx, module_name, pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))
            else:
                module_2 = get_module_by_name(blocks[block_idx], "attn.to_add_out")
                pruner_dict[(block_idx, module_name)] = OBS_Diff_Structured_Joint_Attn(module, module_2, block_idx, args)
                hook_fn = create_hook_fn_Joint_Attn(block_idx, module_name, pruner_dict, timestep_weight)
                hook_fn2 = create_hook_fn_Joint_Attn(block_idx, "attn.to_add_out", pruner_dict, timestep_weight)
                hooks.append(module.register_forward_hook(hook_fn))
                hooks.append(module_2.register_forward_hook(hook_fn2))
        
        # Run prompts in dataloader to collect activations
        print(f"Running diffusion for group {g_idx + 1} to collect activations...")
        # consider batch_size
        batch_size = args.batch_size
        num_batches = (len(dataloader) + batch_size - 1) // batch_size
        for i in range(num_batches):
            prompts = dataloader[i * batch_size:(i + 1) * batch_size]
         
            print(f"  Prompts {i}: {prompts}")
            step_info["current"] = 0
            pipe(
                prompt=prompts,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=["latents"],
                generator=torch.Generator("cuda").manual_seed(args.seed)
            )
        
        # Remove hooks for this group
        for hook in hooks:
            hook.remove()
        
        # Execute pruning for this group
        print(f"Pruning group {g_idx + 1}...")
        for block_idx, module_name in group_modules:
            print(f"Pruning Block {block_idx}: {module_name}")
            sparsity = args.sparsity_ratio[block_idx] if isinstance(args.sparsity_ratio, list) else args.sparsity_ratio
            if module_name == "attn.to_out.0":
                idx_1 = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp,
                    headsize=64
                )
            else:
                idx = pruner_dict[(block_idx, module_name)].struct_prune(
                    sparsity=sparsity,
                    percdamp=args.percdamp
                )
            pruner_dict[(block_idx, module_name)].free()
            
            target_layer = get_module_by_name(blocks[block_idx], module_name)
            
            if module_name == "ff.net.2":
                target_layer_in = get_module_by_name(blocks[block_idx], "ff.net.0.proj")
                id = idx.tolist()
                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, id)
                
            if module_name == "ff_context.net.2":
                target_layer_in = get_module_by_name(blocks[block_idx], "ff_context.net.0.proj")
                id = idx.tolist()
                tp.prune_linear_in_channels(target_layer, id)
                tp.prune_linear_out_channels(target_layer_in, id)

            if module_name == 'attn.to_out.0':
                target_add_layer = get_module_by_name(blocks[block_idx], "attn.to_add_out")
                target_q_layer = get_module_by_name(blocks[block_idx], "attn.to_q")
                target_k_layer = get_module_by_name(blocks[block_idx], "attn.to_k")
                target_v_layer = get_module_by_name(blocks[block_idx], "attn.to_v")
                target_add_k_layer = get_module_by_name(blocks[block_idx], "attn.add_k_proj")
                target_add_q_layer = get_module_by_name(blocks[block_idx], "attn.add_q_proj")
                target_add_v_layer = get_module_by_name(blocks[block_idx], "attn.add_v_proj")
                idx_1 = idx_1.tolist()
                tp.prune_linear_in_channels(target_layer, idx_1)
                tp.prune_linear_in_channels(target_add_layer, idx_1)
                tp.prune_linear_out_channels(target_q_layer, idx_1)
                tp.prune_linear_out_channels(target_k_layer, idx_1)
                tp.prune_linear_out_channels(target_v_layer, idx_1)
                tp.prune_linear_out_channels(target_add_k_layer, idx_1)
                tp.prune_linear_out_channels(target_add_q_layer, idx_1)
                tp.prune_linear_out_channels(target_add_v_layer, idx_1)

                print(f"Previous heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")
                pipe.transformer.transformer_blocks[block_idx].attn.heads -= len(idx_1) // 64
                print(f"Current heads: {pipe.transformer.transformer_blocks[block_idx].attn.heads}")
        torch.cuda.empty_cache()
        print(f"Group {g_idx + 1} pruning completed.")

