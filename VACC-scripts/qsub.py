# accept a timestamp
# read the current date
# pump out a qsub script named by the timestamp, for the current date

import datetime
import sys
import subprocess
import time

jobs = int(subprocess.check_output("showq | grep areagan | wc -l",shell=True))
print jobs

for i in xrange(jobs,800):
    time.sleep(2)
    ctime = subprocess.check_output("date +%S.%M.%H.%d.%m.%y",shell=True).rstrip()

    f = open('currdate.txt','r')
    tmp = f.read().rstrip()
    f.close()

    curr = int(tmp)

    # there are only 24012 gutenberg books
    if curr > 24012:
        break

    # date = datetime.datetime.strptime(tmp,'%Y-%m-%d')
    # date += datetime.timedelta(days=1)

    f = open('currdate.txt','w')
    # float specific okay for int too
    f.write('{0:.0f}'.format(curr+1))
    f.close()

    qsub = '''# This job needs 1 compute node with 1 processor per node.
# It should be allowed to run for up to 30 minutes.
#PBS -l walltime=00:30:00
# Name of job.
#PBS -N gutenbergCrunch
# Join STDERR TO STDOUT.  (omit this if you want separate STDOUT AND STDERR)
#PBS -j oe

cd /users/a/r/areagan/work/2014/2014-09books

python chopuniform.py {0:.0f}

\\rm {1}.qsub

'''.format(curr,ctime)

    # print qsub
    print 'writing {}.qsub'.format(ctime)
    f = open('{}.qsub'.format(ctime),'w')
    f.write(qsub)
    f.close()

    qstatus = subprocess.check_output("qsub {}.qsub".format(ctime),shell=True).rstrip()
    print qstatus







