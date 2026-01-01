import numpy as np
import os
from os.path import join
import argparse
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from logger import init_logger

logger = init_logger('pkt2cell')
CELL_SIZE = 514

def parse_arguments():
    parser = argparse.ArgumentParser(description='Convert packet files (timestamp, size+direction) to cell format (timestamp, direction)')

    # 原有模式参数
    parser.add_argument('--dir',
                        metavar='<traces path>',
                        help='Path to the directory with the cell files.')
    parser.add_argument('--format',
                        metavar='<file suffix>',
                        default="",
                        help='File suffix (empty string for files like 0-2)')
    parser.add_argument('--mon_sites',
                        type=int,
                        metavar='<monitored sites>',
                        default=95,
                        help='Number of monitored sites')
    parser.add_argument('--mon_inst',
                        type=int,
                        metavar='<monitored instances>',
                        default=1000,
                        help='Number of instances per site')
    
    # 新模式参数
    parser.add_argument('--src-dir',
                        metavar='<source traces path>',
                        help='源文件夹路径（包含pkt文件）')
    parser.add_argument('--dst-dir',
                        metavar='<destination path>',
                        help='目标文件夹路径（输出cell文件）')
    
    parser.add_argument('--proc_num',
                        type=int,
                        metavar='<process num>',
                        default=60,
                        help='The num of parallel workers')

    # Parse arguments
    args = parser.parse_args()
    return args


def parse(fpath):
    """原有模式: 保留原文件名"""
    global format, dst_dir
    with open(fpath, "r") as f:
        tmp = f.readlines()
    
    # Parse packet file: column 0 = timestamp, column 1 = size+direction
    t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float)
    pkts = np.array(t)
    
    # Extract filename
    fname = os.path.basename(fpath)
    dst_fname = fname + '.cell'  # Add .cell suffix to output
    
    # Write converted file: split packets into cells based on size
    with open(join(dst_dir, dst_fname), 'w') as f:
        for pkt in pkts:
            cur_t = pkt[0]  # timestamp
            size = pkt[1]   # size + direction
            cur_sign = np.sign(size)  # extract direction
            num_of_cells = int(abs(np.round(size / CELL_SIZE)))  # calculate number of cells
            for _ in range(num_of_cells):
                f.write('{:.6f}\t{:.0f}\n'.format(cur_t, cur_sign))


def parse_with_index(args_tuple):
    """新模式: 按索引重命名"""
    global dst_dir
    fpath, index = args_tuple

    with open(fpath, "r") as f:
        tmp = f.readlines()
    
    # Parse packet file: column 0 = timestamp, column 1 = size+direction
    try:
        t = pd.Series(tmp).str.slice(0, -1).str.split("\t", expand=True).astype(float)
        pkts = np.array(t)
    except Exception as e:
        logger.error(f"解析失败: {fpath}, 错误: {e}")
        return
    
    # 使用索引命名：0.cell, 1.cell, ...
    dst_fname = f'{index}.cell'
    
    # Write converted file: split packets into cells based on size
    with open(join(dst_dir, dst_fname), 'w') as f:
        for pkt in pkts:
            cur_t = pkt[0]  # timestamp
            size = pkt[1]   # size + direction
            cur_sign = np.sign(size)  # extract direction
            num_of_cells = int(abs(np.round(size / CELL_SIZE)))  # calculate number of cells
            for _ in range(num_of_cells):
                f.write('{:.6f}\t{:.0f}\n'.format(cur_t, cur_sign))


if __name__ == '__main__':
    global format, dst_dir
    # parser config and arguments
    args = parse_arguments()
    logger.info("Arguments: %s" % (args))
    
    # 判断使用哪种模式
    if args.src_dir and args.dst_dir:
        # ========== 新模式: 扫描所有文件并按索引重命名 ==========
        logger.info("使用新模式: 扫描所有文件并按索引重命名")
        
        # 扫描源文件夹所有文件
        flist = [os.path.join(args.src_dir, f) 
                 for f in os.listdir(args.src_dir) 
                 if os.path.isfile(os.path.join(args.src_dir, f))]
        flist.sort()  # 排序以保证顺序一致

        logger.info(f"Found {len(flist)} files to convert")

        # Create output directory
        dst_dir = args.dst_dir
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        logger.info(f"Output directory: {dst_dir}")

        # 创建(文件路径, 索引)元组
        flist_with_index = [(fpath, i) for i, fpath in enumerate(flist)]

        # Process files in parallel with progress bar
        logger.info("Starting conversion...")
        with mp.Pool(processes=args.proc_num) as p:
            list(tqdm(p.imap(parse_with_index, flist_with_index), total=len(flist), desc="Converting"))
            p.close()
            p.join()
    
    elif args.dir:
        # ========== 原有模式: 固定命名规则 (i-j) ==========
        logger.info("使用原有模式: 固定命名规则 (i-j)")
        format = args.format
        MON_SITE_NUM = args.mon_sites
        MON_INST_NUM = args.mon_inst

        # Build file list (files without suffix like 0-0, 0-1, etc.)
        flist = []
        for i in range(MON_SITE_NUM):
            for j in range(MON_INST_NUM):
                fpath = os.path.join(args.dir, str(i) + "-" + str(j))
                if os.path.exists(fpath):
                    flist.append(fpath)

        logger.info(f"Found {len(flist)} files to convert")

        # Create output directory
        dst_dir = args.dir.rstrip('/') + '_cell'
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        logger.info(f"Output directory: {dst_dir}")

        # Process files in parallel with progress bar
        logger.info("Starting conversion...")
        with mp.Pool(processes=args.proc_num) as p:
            list(tqdm(p.imap(parse, flist), total=len(flist), desc="Converting"))
            p.close()
            p.join()
    
    else:
        logger.error("请指定参数: --dir (原有模式) 或 --src-dir + --dst-dir (新模式)")
        exit(1)

    logger.info("Conversion completed!")
