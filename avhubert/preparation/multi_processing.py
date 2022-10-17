import math
from multiprocessing import  Process
import os

class BaseLogger:
    def __init__(self, output_dir, headers) -> None:
        self.log_file_name = self.log_file_name = os.path.join(output_dir, 'process_manager_log.csv')
        self._init_log_file(headers)
        
    def _init_log_file(self, headers):
        with open(self.log_file_name, 'w') as fo:
            fo.write(','.join(headers)+'\n')
        print
    
    def write_line(self, values):
        """write a line to csv file created by BaseLogger

        Args:
            values (str): List of string to be written on a new line. List elements MUST be string.
        """
        with open(self.log_file_name, 'a') as fa:
            fa.write(','.join(values)+'\n')
            
    
class BaseProcessManager:
    def __init__(self, root_dir, output_dir, fids, nshard, logger=BaseLogger) -> None:
        """MultiProcessingTaskManager

        Args:
            root_dir (str): input path to dataset like lrs3.
            output_dir (str): output path
            fids (collection): fids collection. 
                If they are not compatible with get functions designed for fids,
                you may override the get functions to suit your need.
            nshard (int): number of shards
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=False)
        self.fids = fids
        self.nshard = nshard
        self.finished = [0] * nshard
        self.process_list = []
        self.num_per_shard = self.get_nums_per_shard(fids, nshard)
        self.is_running = False
        self.logger = logger(output_dir, headers=['finished_rank'])
                  
    def get_nums_per_shard(self, fids, nshard):
        return math.ceil(len(fids)/nshard)
    
    def get_fids_per_shard(self, fids, start_id, end_id):
        return fids[start_id:end_id]
    
    def run(self, Process_class, *extra_args, **extra_kwargs):
        """Run with optional extra args in multi-process manner.
           Warning: You should NOT invoke this method more than once 
           in a same instance of ProcessManager.

        Args:
            Process_class (User-defined process class overriding BaseProcess class): 
                implement your own version of process by overriding __init__ and run
                with any extra arguments in BaseProcess class.
        """
        if self.is_running:
            raise RuntimeError("You need to initialize another Manager instance to carry on more tasks.")
        self.is_running = True
        for rank in range(self.nshard):
            start_id, end_id = self.num_per_shard*rank, self.num_per_shard*(rank+1)
            fids_per_shard = self.get_fids_per_shard(self.fids, start_id, end_id)
            p = self._get_process(Process_class, fids_per_shard, *extra_args, **extra_kwargs)
            p.start()
            self.process_list.append(p)
            print(f"task {rank} submitted :{len(fids_per_shard)} files")
        self.finish()
    
    def finish(self):
        finish_sum = 0
        for rank, p in enumerate(self.process_list):
            p.join()
            finish_sum += 1
            self.finished[rank] = 1
            print(f'{finish_sum} of {self.nshard} processes finished. {self.nshard-finish_sum} to go.')
            print(f'currently finished tasks: {self._get_finished_element_index(1)}')
            self.logger.write_line([str(rank)])
        
        unfinished_idx = self._get_finished_element_index(0)
        if len(unfinished_idx) == 0:
            print(f'ALL {sum(self.finished)} SHARDS ACCOMPLISHED.')
        else:
            print('==========REPORT:UNFINISHED PROCESS==========')
            print(*unfinished_idx)
            
    def _get_process(self, Process_class, fids_per_shard, *args, **kwargs):
        return Process_class(self.root_dir, self.output_dir, fids_per_shard, *args, **kwargs)
            
    def _get_finished_element_index(self, condition_value):
        return [i for i, x in enumerate(self.finished) if x == condition_value]
     
from multiprocessing import Pool
@DeprecationWarning
class PoolProcessManager(BaseProcessManager):
    def __init__(self, root_dir, output_dir, fids, nshard, logger=BaseLogger) -> None:
        super().__init__(root_dir, output_dir, fids, nshard)
        self.process_list = Pool(processes=nshard)
        self.logger = logger(output_dir, ['finished_rank, start, end'])
        
    def run(self, function, *extra_args, **extra_kwargs):
        """Run with optional extra args in multi-process manner.
           Warning: You should NOT invoke this method more than once 
           in a same instance of ProcessManager.

        Args:
            function (User-defined function with at least three input arguments:input_dir, output_dir, fids_per_shard): 
                the return value will be passed to self.callback when finished; feel free to override self.callback to your need.
        """
        if self.is_running:
            raise RuntimeError("You need to initialize another Manager instance to carry on more tasks.")
        self.is_running = True
        for rank in range(self.nshard):
            start_id, end_id = self.num_per_shard*rank, self.num_per_shard*(rank+1)
            fids_per_shard = self.get_fids_per_shard(self.fids, start_id, end_id)
            callback=lambda ret:self.callback(ret, rank, start_id, end_id)
            self.process_list.apply_async(function, args=(self.root_dir, self.output_dir, fids_per_shard, *extra_args,), kwds={**extra_kwargs,}, callback = callback)
            print(f"task {rank} submitted :{len(fids_per_shard)} files")
    
    def finish(self):
        pass
        
    def callback(self, ret, rank, start_id, end_id):
        if ret is not None:
            print(ret)
        self.finished[rank] = 1
        finished = self._get_finished_element_index(1)
        print(f'{rank} finished. {self.nshard - len(finished)} to go. Currently finished:')
        print(*finished)
        self.logger.write_line([str(rank), '%.3f' % (start_id), '%.3f' % (end_id)])
        
    
    
        
class BaseProcess(Process):
    def __init__(self, input_dir, output_dir, fids_per_shard):
        super(BaseProcess,self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fids_per_shard = fids_per_shard
    
    