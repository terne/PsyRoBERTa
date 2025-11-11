import pandas as pd
import numpy as np
import multiprocessing
import queue
import threading
from tqdm import tqdm
from itertools import chain
from collections import OrderedDict
import re
import sys
import os
from preproc_utils import preprocess_wo_NER, lastnames_second_search
import unicodedata
import time
import argparse


def preprocess1(batch, lastnameslst, ind=None, return_dict=None):
    """Removing firstnames and lastnames found by regex. These lastnames are saved and returned. """
    try:
        processed_texts = []
        for loc, text in enumerate(batch.Text.values):
            new_string, firstname_list, lastname_list = preprocess_wo_NER(text,all_names)
            lastnameslst.append(lastname_list)
            #new_string = unicodedata.normalize("NFKD", new_string)
            processed_texts.append(new_string)
        if ind == None:
            return processed_texts
        return_dict[ind] = processed_texts
        # added to test ability to save as each batch is processed
        return ind, processed_texts
    except Exception as e:
        print(f"Error processing batch {ind}: {e}")


def preprocess2(batch, lastnameslst, ind=None, return_dict=None):
    """Extra step of removing lastnames by search for the previously found, 
    that may appear in other contexts not captured by the previous 1st step regex."""
    try:
        processed_texts = []
        for loc, text in enumerate(batch.preproc_1.values):
            new_string_ = lastnames_second_search(text, lastnameslst)
            #new_string_ = unicodedata.normalize("NFKD", new_string_)
            processed_texts.append(new_string_)
        if ind == None:
            return processed_texts
        return_dict[ind] = processed_texts
        #return ind, processed_texts
    except Exception as e:
        print(f"Error processing batch {ind}: {e}")


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--date", type=str, default="231123")
    parser.add_argument("--r", type=int, default=None, help="skiprows")
    parser.add_argument("--n", type=int, default=None, help="nrows")
    parser.add_argument("--part", type=int, default=0, help="The number of processing loop, when processing data in split parts of length nrows (helps with issues of Azure crashing)")

    # r,n, part
    args = parser.parse_args()

    dt = args.date
    skipr = args.r
    if skipr!=None:
        r = range(1,skipr) #skiprows
    else:
        r=skipr
    n = args.n #nrows
    part = args.part

    print(dt)

    totallength=n
    if n==None:
        totallength = 44000000-skipr
    
    # if wanting to split all of the data to each available cpu core at once:
    #chunksize = int(np.ceil(totallenght/multiprocessing.cpu_count())) #2749789
    #otherwise a smaller chunksize is also fine (better):
    chunksize = 2000


    datapath = args.data_path
    print(datapath)
    output_path1= './outputs/clinicalNote44M_{}_preproc1_{}.csv'.format(dt,part)
    rm_lastnames_path = './outputs/clinicalNote44M_{}_removed_lastnames_{}.txt'.format(dt,part)
    output_path2= './outputs/clinicalNote44M_{}_preproc2_{}.csv'.format(dt,part)
    final_output_path = "./outputs/clinicalNote44M_NamesRemoved_{}_{}.csv".format(dt,part)


    # curated list of firstnames
    namelistpath = "large_collected_namelist_edited.txt"
    with open(namelistpath, "r") as fopen:
        all_names = fopen.read().splitlines()


    result = pd.DataFrame()
    #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    jobs = []    
    manager = multiprocessing.Manager()
    lastnameslst = manager.list()
    return_dict = manager.dict()
    
    def my_callback(_):
        # We don't care about the actual result.
        # Just update the progress bar:
        pbar.update(chunksize)
    
    def writer_thread(q, file_lock, output_file):
        while True:
            try:
                index, data = q.get(timeout=3)  # Adjust timeout as needed
                with file_lock:
                    with open(output_file, 'a') as f:  # 'a' for append mode
                        # Assuming 'data' is a list of strings or similar
                        for line in data:
                            line = line.strip("\n")
                            f.write(f"{line}\n")
                q.task_done()
            except queue.Empty:
                break


    with tqdm(total=totallength) as pbar:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            with pd.read_csv(datapath, chunksize=chunksize, usecols=["Text"], skiprows=r, nrows=n) as reader:
                for i, batch in enumerate(reader):
                        
                        p = pool.apply_async(preprocess1, args=(batch, lastnameslst, i, return_dict), callback=my_callback)
                        jobs.append(p)
            
            # Ensure all jobs are finished
            while not all(job.ready() for job in jobs):
                time.sleep(1)  # adjust the sleep time as per your requirement
                
            pool.close()
            pool.join()


    # Check if return_dict is populated
    if not return_dict:
        print("return_dict is empty")


    result["preproc_1"] = list(chain(*list(OrderedDict(sorted(return_dict.items())).values())))

    result.to_csv(output_path1, mode='a', header=not os.path.exists(output_path1))

    lastnameslst = list(chain.from_iterable(lastnameslst))

    with open(rm_lastnames_path, 'a') as fp:
        fp.write('\n'.join(lastnameslst))


   
    print("First part done! Starting second step...")
    result = pd.DataFrame()
    jobs = []    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    with tqdm(total=totallength) as pbar:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            with pd.read_csv(output_path1,chunksize=chunksize) as reader:
                for i, batch in enumerate(reader):
                    p = pool.apply_async(preprocess2, args=(batch, lastnameslst, i, return_dict), callback=my_callback)
                    jobs.append(p)

            # Ensure all jobs are finished
            while not all(job.ready() for job in jobs):
                time.sleep(1)  # adjust the sleep time as per your requirement

            pool.close()
            pool.join()

    # Check if return_dict is populated
    if not return_dict:
        print("return_dict is empty")
    
    result["preproc_2"] = list(chain(*list(OrderedDict(sorted(return_dict.items())).values())))
    result.to_csv(output_path2, mode='a', header=not os.path.exists(output_path2))


    print("Second step done! Now collecting results...")
    datapath1 = datapath
    datapath2 = output_path2

    df1 = pd.read_csv(datapath1, index_col=0, skiprows=r, nrows=n)
    df2 = pd.read_csv(datapath2)
    
    df1["text_names_removed_step2"] = df2.preproc_2.values
    df1.to_csv(final_output_path)

    print("Done!")