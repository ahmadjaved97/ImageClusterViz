"""
Progress tracking utilities.
"""

from tqdm import tqdm

class ProgressTracker:
    """
    Wrapper for progress tracking with optional callbacks.
    """

    def __init__(
        self,
        total,
        desc="Processing",
        disable=False,
        callback=None
    ):
        """
        Initialize progress tracker.
        """

        self.total = total
        self.desc = desc
        self.disable = disable
        self.callback = callback
        self.current = 0

        if not disable:
            self.pbar = tqdm(total = total, desc=desc)
        else:
            self.pbar = None
    
    def update(self, n=1, **kwargs):
        """
        Update progress.
        """

        self.current += n

        if self.pbar is not None:
            self.pbar.update(n)

            if kwargs:
                self.pbar.set_postfix(**kwargs)
        
        if self.callback is not None:
            self.callback(self.current, self.total)
    
    def set_postfix(self, **kwargs):
        """
        Set additional info to display.
        """
        if self.pbar is not None:
            self.pbar.set_postfix(**kwargs)
    
    def close(self):
        """
        Close progress bar.
        """
        if self.pbar is not None:
            self.pbar.close()
    
    def __enter__(self):
        """
        Context manager entry.
        """
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit.
        """
        self.close()
        return False
        