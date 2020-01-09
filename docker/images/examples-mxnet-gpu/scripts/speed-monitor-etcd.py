import os
import logging
import time
import subprocess
import sys
import etcd
import socket

logging.basicConfig(level=logging.INFO,	format='%(asctime)s.%(msecs)03d %(module)s %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
ROLE = os.getenv("ROLE")
WORK_DIR = os.getenv("WORK_DIR")

# read the log file and monitor the training progress
# give log file name
# give record file name
# run the function in a separate thread
def update_speed(logfile, recordfile):
	filesize = 0
	line_number = 0

	"""
	To delete a key in etcd:
	client.delete('/nodes/n1')
	client.delete('/nodes', recursive=True) #this works recursively
	"""
	myname = socket.getfqdn(socket.gethostname(  ))#Get the pod name
	client = etcd.Client(host='10.28.1.1',port=4001)#connect to the etcd cluster	
	key_speed_dir = '/metrics/'+myname+'/speed/avg_speed'#create the key directory of the pod
	client.set(key_speed_dir,'start:')#initial the key value
	key_speed_dir_latest = 'metrics/'+myname+'/speed/avg_speed_latest'#the key directory which will be overwrittened when updated

	# logfile = 'training.log'
	# recordfile = 'speed.txt'	# change to the correct path ....../data/mxnet-data/......
	
	with open(recordfile, 'w') as fh:
		fh.write('0 0\n')
	logging.info('starting speed monitor to track average training speed ...')

	speed_list = []
	disp_interval = 0
	while True:
		time.sleep(5)
		disp_interval += 5
		try:
			cursize = os.path.getsize(logfile)
		except OSError as e:
			logging.warning(e)
			continue
		if cursize == filesize:	# no changes in the log file
			continue
		else:
			filesize = cursize
		
		# Epoch[0] Time cost=50.885
		# Epoch[1] Batch [70]	Speed: 1.08 samples/sec	accuracy=0.000000
		logging.debug("real number of lines" + str(subprocess.check_output("cat " + logfile + " | wc -l", shell=True)))
		with open(logfile, 'r') as f:
			for i in xrange(line_number):
				try:
					f.next()
				except Exception as e:
					logging.error(str(e) + "line_number: " +str(line_number))
			for line in f:
				line_number += 1
				logging.debug("line number: " + str(line_number))
				logging.debug("real number of lines" + str(subprocess.check_output("cat " + logfile + " | wc -l", shell=True)))
				start = line.find('Speed')
				end = line.find('samples')	
				if start > -1 and end > -1 and end > start:
					string = line[start:end].split(' ')[1]
					try:
						speed = float(string)
						speed_list.append(speed)
					except ValueError as e:
						logging.warning(e)
						break
		
		if len(speed_list) == 0:
			continue
			
		avg_speed = sum(speed_list)/len(speed_list)
		# logging.info('Average Training Speed: ' + str(avg_speed))

		#Update the key in etcd
		result = client.read(key_speed_dir)#read the key
		result.value += u' '+str(avg_speed)#append the value
		client.update(result)#update the key
		client.set(key_speed_dir_latest,str(avg_speed))#update the latest key
			
		stb_speed = 0
		if len(speed_list) <= 5:
			stb_speed = avg_speed
		else:
			pos = 2*len(speed_list)/3
			stb_speed = sum(speed_list[pos:])/len(speed_list[pos:])	# only consider the later part

		if disp_interval == 30:
			logging.info('Stable Training Speed: ' + str(stb_speed))
			disp_interval = 0
		
		with open(recordfile, 'w') as fh:
			fh.write(str(avg_speed) + ' ' + str(stb_speed) + '\n')
	
	

def main():
	#logfile = WORK_DIR + 'training.log'
	logfile = '/training.log'
	recordfile =  WORK_DIR + 'speed.txt'
	if ROLE == 'worker':
		update_speed(logfile, recordfile)
	
	
	
if __name__ == '__main__':
	if len(sys.argv) != 1:
		print "Description: monitor training progress in k8s cluster"
		print "Usage: python update_progress.py"
		sys.exit(1)
	main()
