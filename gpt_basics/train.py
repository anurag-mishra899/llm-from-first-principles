import torch
import time
from loader import get_batch

class Trainer:
    def __init__(self,model,train_data,val_data, config, experiment_tag='default'):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.config = config
        self.optimizer = torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
        # history dictionaries to store losses across experiments.
        # structure: { tag1: {'train': [...], 'val': [...]}, tag2: {...}, ... }
        self.history = {}
        # metrics stores timing/throughput measurements per experiment tag
        # structure: { tag1: [ { 'ms_per_step':..., 'tok_per_sec':..., 'step':... }, ... ] }
        self.metrics = {}
        self.tag = None
        self.set_experiment(experiment_tag)

    @torch.no_grad
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train','val']:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x,y = get_batch(self.train_data, self.val_data,
                            self.config.seq_len,self.config.batch_size,
                            self.config.device,split)
                _,loss = self.model(x,y)
                losses[k] = loss
            out[split] = losses.mean()
        self.model.train()
        return out 

    def transformer_matmuls(self,B,T,D,L):

        per_layer = 12*B*T*D*D + 2*B*T*T*D

        total = 3 * L * per_layer

        return total

    def train(self):

        B = self.config.batch_size
        T = self.config.seq_len
        D = self.config.n_embd
        L = self.config.n_layer

        total_time = 0
        total_tokens = 0
        total_matmuls = 0

        for steps in range(self.config.max_iters):

            torch.mps.synchronize()  # use cuda.synchronize() if CUDA
            start = time.perf_counter()

            x,y = get_batch(
                self.train_data,
                self.val_data,
                T,
                B,
                self.config.device,
                'train'
            )

            _, loss = self.model(x,y)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            torch.mps.synchronize()

            end = time.perf_counter()

            step_time = end - start

            # accumulate metrics

            total_time += step_time

            tokens = B*T
            total_tokens += tokens

            total_matmuls += self.transformer_matmuls(B,T,D,L)

            # report at eval interval

            if steps % self.config.eval_interval == 0 and steps > 0:

                losses = self.estimate_loss()
                if self.tag is not None:
                    self.history.setdefault(self.tag, {'train': [], 'val': []})
                    self.history[self.tag]['train'].append(losses['train'].item())
                    self.history[self.tag]['val'].append(losses['val'].item())

                ms_per_step = (total_time / self.config.eval_interval) * 1000

                tokens_per_sec = total_tokens / total_time

                # record metrics under current tag
                if self.tag is not None:
                    self.metrics.setdefault(self.tag, [])
                    self.metrics[self.tag].append({
                        'step': steps,
                        'ms_per_step': float(ms_per_step),
                        'tok_per_sec': float(tokens_per_sec)
                    })

                print(
                    f"step {steps:6d} | "
                    f"train loss {losses['train']:.4f} | "
                    f"val loss {losses['val']:.4f} | "
                    f"{ms_per_step:7.2f} ms/step | "
                    f"{tokens_per_sec:9.0f} tok/s "
                )

                total_time = 0
                total_tokens = 0
                total_matmuls = 0

    # def train(self):
    #     for steps in range(self.config.max_iters):
    #         if steps % self.config.eval_interval==0:
    #             losses = self.estimate_loss()
    #             # record losses under current tag if available
    #             if self.tag is not None:
    #                 self.history.setdefault(self.tag, {'train': [], 'val': []})
    #                 self.history[self.tag]['train'].append(losses['train'].item())
    #                 self.history[self.tag]['val'].append(losses['val'].item())
    #             print(
    #                 f"step: {steps} -> loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}"
    #             )

    #         x,y = get_batch(self.train_data, self.val_data,
    #                         self.config.seq_len,self.config.batch_size,
    #                         self.config.device,'train')
    #         _, loss = self.model(x,y)
    #         self.optimizer.zero_grad(set_to_none=True)
    #         loss.backward()
    #         self.optimizer.step()

    def set_experiment(self, tag):
        """Switch to or initialize a different experiment tag.

        Losses from each evaluation interval will be recorded under the
        corresponding tag. This lets you reuse a single Trainer across
        multiple configurations and then compare histories later.
        """
        self.tag = tag

    def save_history(self, filepath):
        """Persist the full history dictionary to a JSON file.

        The file will contain an object mapping experiment tags to their
        recorded train/val loss lists.
        """
        # if the file already exists, load it and merge the new data so that
        # we don't clobber previous experiments.  This way running multiple
        # configurations in separate notebook sessions will accumulate in a
        # single history file rather than overwriting it each time.
        import json, os

        existing = {}
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            except Exception:
                # if the file is malformed for some reason, ignore it and
                # start fresh (alternatively we could raise)
                existing = {}

        # merge current history into existing; when a tag already exists we
        # extend the lists rather than replace them, allowing multiple runs
        # under the same tag to accumulate sequentially.
        for tag, hist in self.history.items():
            if tag in existing:
                existing[tag].setdefault('train', [])
                existing[tag].setdefault('val', [])
                existing[tag]['train'].extend(hist.get('train', []))
                existing[tag]['val'].extend(hist.get('val', []))
            else:
                existing[tag] = hist.copy()

        with open(filepath, 'w') as f:
            json.dump(existing, f)

    def save_metrics(self, filepath):
        """Save timing/throughput metrics to JSON. Merges with existing file if present."""
        import json, os

        existing = {}
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r') as f:
                    existing = json.load(f)
            except Exception:
                existing = {}

        for tag, metric_list in self.metrics.items():
            if tag in existing:
                existing.setdefault(tag, [])
                existing[tag].extend(metric_list)
            else:
                existing[tag] = list(metric_list)

        with open(filepath, 'w') as f:
            json.dump(existing, f)
