"""
This file is used to generate several job 
scripts and submit all the jobs. In case 
one does not have access to a cpu cluster,
you could also run a single job by making 
appropriate changes.

THIS FILE IS NOT MEANT TO BE RUN DIRECTLY.
"""

import subprocess
import sys

args = sys.argv
img_dir = args[1]
save_dir = args[2]
t_end = float(args[3])
njobs = int(args[4])

print('img_dir')
print(img_dir)

print('save_dir')
print(save_dir)

print('t_end')
print(t_end)


def generate(n1,n2, save_dir, img_dir, t_end):
    """Generates a pbs script"""
    
    with open('submit.pbs', 'w') as f:
        f.write("""
            #!/bin/bash
            #PBS -q secondary
            #PBS -k o
            #PBS -l nodes=1:ppn=1,walltime=4:00:00
            #PBS -M kkothar3@illinois.edu
            #PBS -N rtm
            #PBS -j oe
            cd /home/kkothar3/FIO
            module load matlab/9.5
            mkdir -p %s
            matlab -nodesktop -nosplash -nodisplay -r "save_dir='%s'; n1=%d; n2=%d; t_end=%f; img_dir='%s'; run kwave_rtc.m"
        """%(save_dir, save_dir, n1, n2, t_end, img_dir))
        f.close()
    
    return

def loop(n, offset=0):
    r = 10
    for i in range(n):
        ## generate a script for first 50 sensor_traces
        generate(offset+i*r,offset+(i+1)*r-1, save_dir, img_dir, t_end)

        exit_status = subprocess.call("""qsub -p 1020 submit.pbs""", shell=True)
        if exit_status == 1:
            print("Failed to submit job %d"%i)
            
    return 


if __name__ == "__main__":
    loop(njobs, 0)
