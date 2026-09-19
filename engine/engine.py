"""Qwen3 4B engine: native prefill, CUDA-graph decode over a static KV cache.

Prefill is the untouched Transformers path, so TTFT and the first token match
native exactly. Decode then runs a hand-written single-token step over the
loaded layers, with a preallocated KV cache and no host work per step, captured
once into a CUDA graph during warmup and replayed for every later token. If the
fast path cannot be set up, the engine falls back to the native decode loop.
"""

import sys
import traceback

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import rotate_half


class DecodeRunner:
    """Single-token decode step for a fixed batch size and cache capacity."""

    def __init__(self, model, batch: int, capacity: int, use_graph: bool) -> None:
        cfg = model.config
        base = model.model
        weight = base.embed_tokens.weight
        device, dtype = weight.device, weight.dtype

        self.model = model
        self.batch = batch
        self.capacity = capacity
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.group = self.n_heads // self.n_kv
        self.scale = self.head_dim ** -0.5

        shape = (batch, self.n_kv, capacity, self.head_dim)
        n_layers = len(base.layers)
        self.k_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.v_cache = [torch.zeros(shape, dtype=dtype, device=device) for _ in range(n_layers)]
        self.tok = torch.zeros(batch, dtype=torch.int64, device=device)
        self.pos = torch.zeros(1, dtype=torch.int64, device=device)
        self.pos_ids = self.pos.view(1, 1)
        self.arange = torch.arange(capacity, dtype=torch.int64, device=device)

        # One matmul for q/k/v and one for gate/up per layer, built once here.
        self.qkv_weight = [
            torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight, l.self_attn.v_proj.weight], dim=0)
            for l in base.layers
        ]
        self.gate_up_weight = [
            torch.cat([l.mlp.gate_proj.weight, l.mlp.up_proj.weight], dim=0) for l in base.layers
        ]
        self.q_dim = self.n_heads * self.head_dim
        self.kv_dim = self.n_kv * self.head_dim
        self.inter = cfg.intermediate_size

        self.graph = None
        if use_graph:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()

    def _step(self) -> None:
        """Consume self.tok at position self.pos; write the next token to self.tok."""
        model = self.model
        base = model.model
        B, hd = self.batch, self.head_dim

        x = base.embed_tokens(self.tok)
        cos, sin = base.rotary_emb(x, self.pos_ids)

        bias = torch.zeros(self.capacity, dtype=x.dtype, device=x.device)
        bias.masked_fill_(self.arange > self.pos, float("-inf"))
        mask = bias.view(1, 1, 1, self.capacity).expand(B, self.n_kv, self.group, self.capacity)

        for i, layer in enumerate(base.layers):
            attn = layer.self_attn
            h = layer.input_layernorm(x)
            qkv = F.linear(h, self.qkv_weight[i])
            q = attn.q_norm(qkv[:, : self.q_dim].view(B, self.n_heads, hd))
            k = attn.k_norm(qkv[:, self.q_dim : self.q_dim + self.kv_dim].view(B, self.n_kv, hd))
            v = qkv[:, self.q_dim + self.kv_dim :].view(B, self.n_kv, 1, hd)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            self.k_cache[i].index_copy_(2, self.pos, k.unsqueeze(2))
            self.v_cache[i].index_copy_(2, self.pos, v)

            # Query heads h*group..h*group+group-1 share KV head h, so the group
            # becomes the query axis and no K/V expansion is needed.
            o = F.scaled_dot_product_attention(
                q.view(B, self.n_kv, self.group, hd),
                self.k_cache[i],
                self.v_cache[i],
                attn_mask=mask,
                scale=self.scale,
            )
            x = x + attn.o_proj(o.reshape(B, self.n_heads * hd))
            mlp = layer.mlp
            gate_up = F.linear(layer.post_attention_layernorm(x), self.gate_up_weight[i])
            x = x + mlp.down_proj(mlp.act_fn(gate_up[:, : self.inter]) * gate_up[:, self.inter :])

        logits = model.lm_head(base.norm(x))
        self.tok.copy_(logits.argmax(dim=-1))
        self.pos.add_(1)

    def load(self, cache, first_tokens: torch.Tensor, prompt_len: int) -> None:
        """Adopt a prefilled cache; first_tokens are the ids to consume next."""
        for i in range(len(self.k_cache)):
            k, v = cache[i]
            self.k_cache[i][:, :, :prompt_len].copy_(k)
            self.v_cache[i][:, :, :prompt_len].copy_(v)
        self.tok.copy_(first_tokens)
        self.pos.fill_(prompt_len)

    def step(self) -> list[int]:
        if self.graph is not None:
            self.graph.replay()
        else:
            self._step()
        return self.tok.tolist()


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        self._runner = None
        self._runner_key = None

    def _runner_for(self, batch: int, prompt_len: int, max_new_tokens: int) -> DecodeRunner:
        capacity = ((prompt_len + max_new_tokens + 63) // 64) * 64
        key = (batch, capacity)
        if self._runner is None or self._runner_key != key:
            self._runner = None
            self._runner_key = None
            torch.cuda.empty_cache()
            self._runner = DecodeRunner(self.model, batch, capacity, use_graph=True)
            self._runner_key = key
        return self._runner

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if max_new_tokens <= 0:
            return
        current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        batch, prompt_len = current.shape
        with torch.inference_mode():
            output = self.model(
                input_ids=current,
                past_key_values=None,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values
            del output
            yield current[:, 0].tolist()

            remaining = max_new_tokens - 1
            if remaining == 0:
                return

            runner = None
            try:
                runner = self._runner_for(batch, prompt_len, max_new_tokens)
                runner.load(cache, current[:, 0], prompt_len)
            except Exception:
                print(
                    "[engine] fast decode unavailable, using native decode:\n"
                    + traceback.format_exc(),
                    file=sys.stderr,
                    flush=True,
                )
                runner = None
                self._runner = None
                self._runner_key = None

            if runner is not None:
                del cache
                for _ in range(remaining):
                    yield runner.step()
                return

            for _ in range(remaining):
                output = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()
