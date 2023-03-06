import argparse 
import os

def gen_block(batch_size, model_size, device, mode, store_dir):
    dmap = {"CPU": "CPU", "GPU": "GPU:0"}
    lines = []
    for workload in ["tree_lstm_internal", "tree_lstm_leaf", "lstm", "gru", "tree_gru_internal", "tree_gru_leaf", "mvrnn"]:
        cmd = f"./test_block {workload} {device} {batch_size} {model_size} {mode} {store_dir} --dynet-autobatch 1 --dynet-devices {dmap[device]} "
        cmds = [cmd, f"if ! (($? == 0)); then", f"\techo \"{cmd}\" >> error.txt", "fi"]
        lines.extend(cmds)
        
    with open("run_block.sh", "a") as f:
        f.write('\n'.join(lines) + '\n')
    

def gen_performance(batch_size, model_sizes, device, mode, store_dir, workload):
    lines = []
    dmap = {"CPU": "CPU", "GPU": "GPU:0"}
    for model_size in model_sizes:
        cmd = f'./test_graph {workload} {device} {batch_size} {model_size} {mode} {store_dir} --dynet-autobatch 1 --dynet-devices {dmap[device]}'
        cmds = [cmd, f"if ! (($? == 0)); then", f"\techo \"{cmd}\" >> error.txt", "fi"]
        lines.extend(cmds)

    filename = f'./sh/run_{device}-{workload}-{batch_size}-{mode}.sh'
    print('write to ', filename)
    with open(filename, "w") as f:
        f.write('\n'.join(lines))
    with open('run.sh', 'a') as f:
        f.write(f'source {filename}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, nargs = '+', required = True)
    parser.add_argument("--model_size", type = int, nargs = '+', required = True)
    parser.add_argument('--device', type = str, nargs = '+', required = True)
    parser.add_argument('--mode', type=int, choices = [0,1], default = 0, help='\
        0: test performance;\n\
        1: test compile time;')
    parser.add_argument('--block', action = 'store_true')
    parser.add_argument('--store_dir', default = 'tmp', type = str)
    parser.add_argument('--workload', choices = ["bilstm_tagger", "bilstm_tagger_withchar", "tree_lstm", "tree_gru", "mvrnn", "double_typed_treenn", "lattice_lstm", "lattice_gru", 'lstm_nmt'], nargs = '+')
    
    args = parser.parse_args()
    
    args.store_dir += '/'
    
    os.system('mkdir -p sh')
    
    if not args.block:
        with open('run.sh', 'w') as f:
            f.write('rm error.txt\n')
            f.write('touch error.txt\n')
    else:
        os.system("rm run_block.sh") 
        with open('run_block.sh', 'w') as f:
            f.write('rm error.txt\n')
            f.write('touch error.txt\n')
        print (f'sh is run_block.sh, error cmds in error.txt')


    if args.block:
        for b in args.batch_size:
            for h in args.model_size:
                for d in args.device:
                    gen_block(b, h, d, args.mode, args.store_dir)
    else:
        for b in args.batch_size:
            for workload in args.workload:
                for d in args.device:
                    gen_performance(b, args.model_size, d, args.mode, args.store_dir, workload)