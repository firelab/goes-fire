import queue
import threading

class ThreadManager(object)  :
    """Creates and manages threads and queues for bidirectional 
       communications with the worker."""

    def __init__(self, worker, collector=None, num_threads=4, killsig=None) : 
        self.worker = worker
        self.collector = collector
        self.killsig = killsig
        self.num_threads = num_threads
        self.threads = []
        self.work = queue.Queue()
        self.product = queue.Queue()
        self.collected_products = [] 


    def start(self) : 
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker)
            t.start()
            self.threads.append(t)

    @property
    def empty(self)  : 
        """true if both work and product queues are empty."""
        return self.work.empty() and self.product.empty()

    def reset_products(self) : 
        self.collected_products = [] 


    def collect(self) : 
        """collects results from the product queue until 
        both queues are empty."""
        while not self.empty : 
            if self.collector is not None : 
                self.collector(self.product.get())
            else : 
                self.collected_products.append(self.product.get())
        

    def kill(self, block=True)  :
        """sends the "killsig" object to all the threads"""
        for i in range(self.num_threads) : 
            self.work.put(self.killsig)
        if block : 
            for t in self.threads : 
                t.join()
        
        
