import subprocess
import time
import os; rootdir = os.path.split(__file__)[0]

def gpcc_encode(filedir, bin_dir, posQuantscale=1, tmc3dir='tmc3_v14', cfgdir='dense.cfg', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cfgdir = os.path.join(rootdir, cfgdir)
    cmd = tmc3dir + ' --mode=0 ' \
        + ' --config='+cfgdir \
        + ' --positionQuantizationScale='+str(posQuantscale) \
        + ' --uncompressedDataPath='+filedir \
        + ' --compressedStreamPath='+bin_dir
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: print_log(subp)
    # headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    # results = {}
    # c=subp.stdout.readline()
    # while c:
    #     if DBG: print(c)
    #     line = c.decode(encoding='utf-8')
    #     for _, key in enumerate(headers):
    #         if line.find(key) != -1: 
    #             value = number_in_line(line)
    #             results[key] = value
    #     c=subp.stdout.readline()

    return subp

def gpcc_decode(bin_dir, dec_dir,  tmc3dir='tmc3_v14', DBG=False):
    tmc3dir = os.path.join(rootdir, tmc3dir)
    cmd = tmc3dir + ' --mode=1 ' \
        + ' --compressedStreamPath='+bin_dir \
        + ' --reconstructedDataPath='+dec_dir \
        + ' --outputBinaryPly=0'
    subp = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    subp.wait()
    if DBG: print_log(subp)
    # headers = ['Total bitstream size', 'Processing time (user)', 'Processing time (wall)']
    # results = {}
    # c=subp.stdout.readline()
    # while c:
    #     if show: print(c)   
    #     line = c.decode(encoding='utf-8')
    #     for _, key in enumerate(headers):
    #         if line.find(key) != -1: 
    #             value = number_in_line(line)
    #             results[key] = value   
    #     c=subp.stdout.readline()

    return subp

def number_in_line(line):
    wordlist = line.split(' ')
    for _, item in enumerate(wordlist):
        try: number = float(item) 
        except ValueError: continue
        
    return number

def print_log(p):
    c=p.stdout.readline()
    while c:
        print(c)
        c=p.stdout.readline()
        
    return 