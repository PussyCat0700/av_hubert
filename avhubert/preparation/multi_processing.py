import math
from multiprocessing import  Process
class BaseProcessManager:
    def __init__(self, root_dir, output_dir, fids, nshard) -> None:
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
        self.fids = fids
        self.nshard = nshard
        self.finished = [0] * nshard
        self.process_list = []
        self.num_per_shard = self.get_nums_per_shard(fids, nshard)
        self.is_running = False
        
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
        
    
    
        
class BaseProcess(Process):
    def __init__(self, input_dir, output_dir, fids_per_shard):
        super(BaseProcess,self).__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.fids_per_shard = fids_per_shard
    
        
    
    
"""
test
"""
class MyProcess(BaseProcess):
    def __init__(self, input_dir, output_dir, fids_per_shard, params):
        super().__init__(input_dir, output_dir, fids_per_shard)
        self.params = params
    
    def run(self):
        print(f'{self.params}')
        
if __name__=='__main__':
    manager = BaseProcessManager('/path/to/input', '/path/to/output', [1,2,3,4,5], 2)
    manager.run(MyProcess, 2)
    manager = BaseProcessManager('/path/to/input', '/path/to/output', [1,2,3,4,5], 2)
    manager.run(BaseProcess)
    