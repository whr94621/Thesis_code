from gzip import GzipFile
import sys
import logging
import numpy.random as random
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
                    level=logging.INFO)

raw_corpus = sys.argv[1]
new_corpus = sys.argv[2]
preportion = float(sys.argv[3])

with GzipFile(raw_corpus,'r') as fin, open(new_corpus,'w') as fout:
    buffer_size = 5 * 1024 * 1024
    raw_id = 0
    new_id = 0
    report_interval = 100000
    report_threshold =100000
    
    lines = fin.readlines(buffer_size)
    while lines:
        for line in lines:
            idx = random.choice([0,1],p=[preportion,1-preportion])
            if idx == 0:
                new_id += 1
                fout.write(line)
            raw_id += 1
        lines = fin.readlines(buffer_size)
        if raw_id > report_threshold:    
            logging.info('Have selected %d out of %d passages' % (new_id, raw_id))
            report_threshold += report_interval           