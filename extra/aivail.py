# Copyright (c) 2024 James Chapman
#
# This software is dual-licensed:
#
# - For individuals and non-commercial use: Licensed under the MIT License.
# - For commercial or corporate use: A separate commercial license is required.
#
# To obtain a commercial license, please contact: iconoclastdao@gmail.com
#
# By using this software, you agree to these terms.
#
# MIT License (for individuals and non-commercial use):
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import time
import hashlib
import base64
import argparse
import struct
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

class TCCLogger:
    def __init__(self):
        self.tcc_log: List[TCCLogEntry] = []
        self.step_counter: int = 0

    def log(self, operation: str, input_data: bytes, output_data: bytes, metadata: Dict[str, Any] = None, log_level: str = "INFO", error_code: str = "NONE") -> None:
        entry = TCCLogEntry(self.step_counter, operation, input_data, output_data, metadata or {}, log_level, error_code, prev_hash=self._compute_prev_hash())
        self.tcc_log.append(entry)
        self.step_counter += 1

    def _compute_prev_hash(self) -> bytes:
        if not self.tcc_log:
            return b'\x00' * 32
        last_entry = self.tcc_log[-1]
        return hashlib.sha256(last_entry.to_bytes()).digest()

    def save_log(self, filename: str) -> None:
        with open(filename, 'w') as f:
            for entry in self.tcc_log:
                f.write(json.dumps(entry.to_json()) + '\n')

class TCCLogEntry:
    def __init__(self, step: int, operation: str, input_data: bytes, output_data: bytes, metadata: Dict[str, Any], log_level: str, error_code: str, prev_hash: bytes):
        self.step = step
        self.operation = operation
        self.input_data = input_data
        self.output_data = output_data
        self.metadata = metadata
        self.log_level = log_level
        self.error_code = error_code
        self.prev_hash = prev_hash
        self.operation_id = hashlib.sha256(f"{step}:{operation}:{time.time_ns()}".encode()).hexdigest()[:32]
        self.timestamp = time.time_ns()
        self.execution_time_ns = 0

    def to_bytes(self) -> bytes:
        start_time = time.time_ns()
        step_bytes = struct.pack('>I', self.step)
        op_bytes = self.operation.encode('utf-8').ljust(32, b'\x00')[:32]
        input_len = len(self.input_data)
        output_len = len(self.output_data)
        input_len_bytes = struct.pack('>I', input_len)
        output_len_bytes = struct.pack('>I', output_len)
        meta_bytes = json.dumps(self.metadata).encode('utf-8').ljust(128, b'\x00')[:128]
        level_bytes = self.log_level.encode('utf-8').ljust(16, b'\x00')[:16]
        error_bytes = self.error_code.encode('utf-8').ljust(16, b'\x00')[:16]
        op_id_bytes = self.operation_id.encode('utf-8').ljust(32, b'\x00')[:32]
        ts_bytes = struct.pack('>Q', self.timestamp)
        exec_time_bytes = struct.pack('>Q', self.execution_time_ns)
        result = (step_bytes + op_bytes + input_len_bytes + self.input_data +
                  output_len_bytes + self.output_data + meta_bytes + level_bytes +
                  error_bytes + self.prev_hash + op_id_bytes + ts_bytes + exec_time_bytes)
        self.execution_time_ns = time.time_ns() - start_time
        return result

    def to_json(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "operation": self.operation,
            "input_data": base64.b64encode(self.input_data).decode('utf-8'),
            "output_data": base64.b64encode(self.output_data).decode('utf-8'),
            "metadata": self.metadata,
            "log_level": self.log_level,
            "error_code": self.error_code,
            "prev_hash": base64.b64encode(self.prev_hash).decode('utf-8'),
            "operation_id": self.operation_id,
            "timestamp": self.timestamp,
            "execution_time_ns": self.execution_time_ns
        }

class LLMModule(ABC):
    def __init__(self):
        self.logger = TCCLogger()

    @abstractmethod
    def compute(self, input_data: Any) -> Any:
        pass

    @abstractmethod
    def reverse(self, output_data: Any) -> Any:
        pass

    @abstractmethod
    def mimic_transformation(self, input_data: Any, ref_input: Any, ref_output: Any) -> Any:
        pass

class ModelManager:
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

class TokenizerModule(LLMModule):
    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.tokenizer = model_manager.tokenizer

    def compute(self, input_text: str) -> Dict[str, Any]:
        start_time = time.time_ns()
        tokens = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        token_ids = tokens["input_ids"].numpy().tobytes()
        attention_mask = tokens["attention_mask"].numpy().tobytes()
        output = {"input_ids": token_ids, "attention_mask": attention_mask}
        self.logger.log(
            "tokenize",
            input_text.encode('utf-8'),
            json.dumps(output).encode('utf-8'),
            {"token_count": len(tokens["input_ids"][0]), "execution_time_ns": time.time_ns() - start_time}
        )
        return output

    def reverse(self, output_data: Dict[str, Any]) -> str:
        start_time = time.time_ns()
        input_ids = np.frombuffer(output_data["input_ids"], dtype=np.int64).reshape(1, -1)
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        self.logger.log(
            "reverse_tokenize",
            json.dumps(output_data).encode('utf-8'),
            text.encode('utf-8'),
            {"execution_time_ns": time.time_ns() - start_time}
        )
        return text

    def mimic_transformation(self, input_text: str, ref_input: str, ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        adjusted_text = input_text[:len(ref_input)] if len(input_text) > len(ref_input) else input_text + " " * (len(ref_input) - len(input_text))
        output = self.compute(adjusted_text)
        self.logger.log(
            "mimic_tokenize",
            input_text.encode('utf-8'),
            json.dumps(output).encode('utf-8'),
            {"ref_input": ref_input, "adjusted_text": adjusted_text}
        )
        return output

class EmbedderModule(LLMModule):
    def __init__(self, model_manager: ModelManager):
        super().__init__()
        self.model = model_manager.model
        self.embedding_layer = self.model.get_input_embeddings()

    def compute(self, tokens: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        input_ids = torch.from_numpy(np.frombuffer(tokens["input_ids"], dtype=np.int64).reshape(1, -1))
        embeddings = self.embedding_layer(input_ids).detach().numpy().tobytes()
        output = {"embeddings": embeddings, "input_ids": tokens["input_ids"]}
        self.logger.log(
            "embed",
            tokens["input_ids"],
            embeddings,
            {"shape": str(np.frombuffer(embeddings).shape), "execution_time_ns": time.time_ns() - start_time}
        )
        return output

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        input_ids = output_data["input_ids"]
        self.logger.log(
            "reverse_embed",
            output_data["embeddings"],
            input_ids,
            {"execution_time_ns": time.time_ns() - start_time}
        )
        return {"input_ids": input_ids}

    def mimic_transformation(self, tokens: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        output = self.compute(tokens)
        self.logger.log(
            "mimic_embed",
            tokens["input_ids"],
            output["embeddings"],
            {"ref_input_ids": ref_input["input_ids"].hex()}
        )
        return output

class TransformerLayerModule(LLMModule):
    def __init__(self, model_manager: ModelManager, layer_idx: int = 0):
        super().__init__()
        self.model = model_manager.model
        self.layer = self.model.transformer.h[layer_idx]
        self.layer_idx = layer_idx

    def compute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        embeddings = torch.from_numpy(np.frombuffer(input_data["embeddings"]).reshape(1, -1, self.model.config.hidden_size))
        input_ids = input_data["input_ids"]
        attention_mask = torch.from_numpy(np.frombuffer(input_data.get("attention_mask", b""), dtype=np.int64).reshape(1, -1))
        outputs = self.layer(embeddings, attention_mask=attention_mask)[0].detach().numpy().tobytes()
        output = {"hidden_states": outputs, "input_ids": input_ids}
        self.logger.log(
            "transformer_layer",
            input_data["embeddings"],
            outputs,
            {"layer_idx": self.layer_idx, "execution_time_ns": time.time_ns() - start_time}
        )
        return output

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        input_ids = output_data["input_ids"]
        self.logger.log(
            "reverse_transformer_layer",
            output_data["hidden_states"],
            input_ids,
            {"layer_idx": self.layer_idx, "execution_time_ns": time.time_ns() - start_time}
        )
        return {"embeddings": output_data["hidden_states"], "input_ids": input_ids}

    def mimic_transformation(self, input_data: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        output = self.compute(input_data)
        self.logger.log(
            "mimic_transformer_layer",
            input_data["embeddings"],
            output["hidden_states"],
            {"layer_idx": self.layer_idx, "ref_input_ids": ref_input["input_ids"].hex()}
        )
        return output

class DecoderModule(LLMModule):
    def __init__(self, model_manager: ModelManager, temperature: float = 1.0, top_k: int = 50):
        super().__init__()
        self.model = model_manager.model
        self.tokenizer = model_manager.tokenizer
        self.temperature = temperature
        self.top_k = top_k

    def compute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        hidden_states = torch.from_numpy(np.frombuffer(input_data["hidden_states"]).reshape(1, -1, self.model.config.hidden_size))
        input_ids = torch.from_numpy(np.frombuffer(input_data["input_ids"], dtype=np.int64).reshape(1, -1))
        logits = self.model.lm_head(hidden_states).detach()
        probs = torch.softmax(logits / self.temperature, dim=-1)
        top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1)
        token_idx = torch.multinomial(top_k_probs[0, -1], 1).item()
        next_token = top_k_indices[0, -1, token_idx].item()
        output_text = self.tokenizer.decode([next_token], skip_special_tokens=True)
        output = {"output_text": output_text, "logits": logits.numpy().tobytes(), "next_token": next_token}
        self.logger.log(
            "decode",
            input_data["hidden_states"],
            output_text.encode('utf-8'),
            {"next_token": next_token, "execution_time_ns": time.time_ns() - start_time}
        )
        return output

    def reverse(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        logits = np.frombuffer(output_data["logits"]).reshape(1, -1, self.model.config.vocab_size)
        hidden_states = self.model.lm_head.weight.data.T @ torch.from_numpy(logits).squeeze(0).T
        self.logger.log(
            "reverse_decode",
            output_data["output_text"].encode('utf-8'),
            hidden_states.numpy().tobytes(),
            {"execution_time_ns": time.time_ns() - start_time}
        )
        return {"hidden_states": hidden_states.numpy().tobytes(), "input_ids": output_data.get("input_ids", b"")}

    def mimic_transformation(self, input_data: Dict[str, Any], ref_input: Dict[str, Any], ref_output: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time_ns()
        output = self.compute(input_data)
        self.logger.log(
            "mimic_decode",
            input_data["hidden_states"],
            output["output_text"].encode('utf-8'),
            {"ref_output_text": ref_output["output_text"]}
        )
        return output

class LLMEntropyEngine:
    FEE_PER_SAMPLING = 1000

    def __init__(self, initial_fee: int = 1000):
        self.fee_per_sampling = initial_fee
        self.commitments = {}
        self.commitment_timestamps = {}
        self.logger = TCCLogger()
        self.random_state = random.Random()

    def commit_sampling(self, user_id: str, seed: int, temperature: float) -> None:
        commitment = hashlib.sha256(f"{seed}:{temperature}".encode()).digest()
        if user_id in self.commitments:
            raise ValueError("Commitment already exists for user")
        self.commitments[user_id] = (seed, temperature)
        self.commitment_timestamps[user_id] = time.time_ns()
        self.logger.log(
            "sampling_committed",
            commitment,
            b"",
            {"user_id": user_id, "seed": seed, "temperature": temperature}
        )

    def reveal_sampling(self, user_id: str, seed: int, temperature: float, fee: int) -> None:
        if fee < self.fee_per_sampling:
            raise ValueError("Insufficient fee")
        if user_id not in self.commitments:
            raise ValueError("No commitment found")
        if self.commitments[user_id] != (seed, temperature):
            raise ValueError("Invalid commitment")
        if (time.time_ns() - self.commitment_timestamps[user_id]) > 86400 * 1_000_000_000:
            raise ValueError("Commitment expired")
        del self.commitments[user_id]
        del self.commitment_timestamps[user_id]
        self.random_state.seed(seed)
        self.logger.log(
            "sampling_revealed",
            f"{seed}:{temperature}".encode(),
            b"",
            {"user_id": user_id}
        )

    def save_log(self, filename: str) -> None:
        self.logger.save_log(filename)

class LLMCoordinator:
    def __init__(self, engine_a: LLMEntropyEngine, engine_b: LLMEntropyEngine, engine_c: LLMEntropyEngine):
        if not all([engine_a, engine_b, engine_c]) or len(set([id(engine_a), id(engine_b), id(engine_c)])) != 3:
            raise ValueError("Invalid or duplicate engines")
        self.engine_a = engine_a
        self.engine_b = engine_b
        self.engine_c = engine_c
        self.logger = TCCLogger()

    def commit_sampling_all(self, user_id: str, seed_a: int, temp_a: float, seed_b: int, temp_b: float, seed_c: int, temp_c: float) -> None:
        self.engine_a.commit_sampling(user_id, seed_a, temp_a)
        self.engine_b.commit_sampling(user_id, seed_b, temp_b)
        self.engine_c.commit_sampling(user_id, seed_c, temp_c)
        self.logger.log(
            "commit_sampling_all",
            f"{seed_a}:{temp_a}:{seed_b}:{temp_b}:{seed_c}:{temp_c}".encode(),
            b"",
            {"user_id": user_id}
        )

    def reveal_sampling_all(self, user_id: str, seed_a: int, temp_a: float, seed_b: int, temp_b: float, seed_c: int, temp_c: float, fee: int) -> None:
        total_fee = 0
        has_a = user_id in self.engine_a.commitments
        has_b = user_id in self.engine_b.commitments
        has_c = user_id in self.engine_c.commitments
        if has_a:
            total_fee += self.engine_a.fee_per_sampling
        if has_b:
            total_fee += self.engine_b.fee_per_sampling
        if has_c:
            total_fee += self.engine_c.fee_per_sampling
        if fee < total_fee:
            raise ValueError("Insufficient fee")
        if has_a:
            self.engine_a.reveal_sampling(user_id, seed_a, temp_a, self.engine_a.fee_per_sampling)
        if has_b:
            self.engine_b.reveal_sampling(user_id, seed_b, temp_b, self.engine_b.fee_per_sampling)
        if has_c:
            self.engine_c.reveal_sampling(user_id, seed_c, temp_c, self.engine_c.fee_per_sampling)
        self.logger.log(
            "reveal_sampling_all",
            f"{seed_a}:{temp_a}:{seed_b}:{temp_b}:{seed_c}:{temp_c}".encode(),
            b"",
            {"user_id": user_id, "total_fee": total_fee}
        )

    def save_log(self, filename: str) -> None:
        self.logger.save_log(filename)

class LLMFlow:
    def __init__(self, steps: List[Tuple[str, LLMModule, Dict[str, Any]]], reference_input: str):
        self.steps = steps
        self.logger = TCCLogger()
        self.flow_log: List[Dict[str, Any]] = []
        self.reference_input = reference_input
        self.reference_outputs = self._compute_reference_outputs()

    def _compute_reference_outputs(self) -> List[Any]:
        outputs = [self.reference_input]
        current_data = self.reference_input
        for step_name, module, _ in self.steps:
            output_data = module.compute(current_data)
            outputs.append(output_data)
            current_data = output_data
        return outputs

    def execute(self, input_text: str) -> str:
        current_data = input_text
        start_time = time.time_ns()
        for step_idx, (step_name, module, params) in enumerate(self.steps):
            output_data = module.compute(current_data)
            output_bytes = json.dumps(output_data).encode('utf-8') if isinstance(output_data, dict) else output_data.encode('utf-8')
            input_bytes = current_data.encode('utf-8') if isinstance(current_data, str) else json.dumps(current_data).encode('utf-8')
            metadata = {
                "step_index": step_idx,
                "step_name": step_name,
                "params": params,
                "output_type": str(type(output_data))
            }
            self.logger.log(
                f"flow_{step_name}",
                input_bytes,
                output_bytes,
                metadata
            )
            self.flow_log.append({
                "step_index": step_idx,
                "step_name": step_name,
                "operation_id": self.logger.tcc_log[-1].operation_id,
                "input_data": base64.b64encode(input_bytes).decode('utf-8'),
                "output_data": base64.b64encode(output_bytes).decode('utf-8'),
                "timestamp": time.time_ns()
            })
            current_data = output_data
        final_output = current_data["output_text"] if isinstance(current_data, dict) else current_data
        self.logger.log(
            "flow_complete",
            input_text.encode('utf-8'),
            final_output.encode('utf-8'),
            {"total_steps": len(self.steps), "execution_time_ns": time.time_ns() - start_time}
        )
        return final_output

    def reverse(self, target_output: str) -> str:
        start_time = time.time_ns()
        target_bytes = target_output.encode('utf-8')
        for entry in reversed(self.logger.tcc_log):
            if entry.operation == "flow_complete" and entry.output_data == target_bytes:
                current_output = {"output_text": target_output}
                for step_idx in range(len(self.steps) - 1, -1, -1):
                    step_name, module, _ = self.steps[step_idx]
                    current_input = module.reverse(current_output)
                    input_bytes = current_input.encode('utf-8') if isinstance(current_input, str) else json.dumps(current_input).encode('utf-8')
                    output_bytes = json.dumps(current_output).encode('utf-8') if isinstance(current_output, dict) else current_output.encode('utf-8')
                    self.logger.log(
                        f"reverse_{step_name}",
                        output_bytes,
                        input_bytes,
                        {"step_index": step_idx}
                    )
                    current_output = current_input
                final_input = current_output if isinstance(current_output, str) else self.steps[0][1].reverse(current_output)
                self.logger.log(
                    "reverse_complete",
                    target_bytes,
                    final_input.encode('utf-8'),
                    {"reconstructed_input": final_input}
                )
                return final_input
        self.logger.log(
            "reverse",
            target_bytes,
            b"",
            {"error": "Target output not found"},
            "ERROR",
            "NOT_FOUND"
        )
        raise ValueError("Target output not found in log")

    def reverse_arbitrary(self, target_output: str, arbitrary_input: str) -> str:
        start_time = time.time_ns()
        current_data = arbitrary_input
        for step_idx in range(len(self.steps)):
            step_name, module, _ = self.steps[step_idx]
            ref_input = self.reference_outputs[step_idx]
            ref_output = self.reference_outputs[step_idx + 1]
            current_data = module.mimic_transformation(current_data, ref_input, ref_output)
        current_output = {"output_text": target_output}
        for step_idx in range(len(self.steps) - 1, -1, -1):
            step_name, module, _ = self.steps[step_idx]
            current_input = module.reverse(current_output)
            input_bytes = current_input.encode('utf-8') if isinstance(current_input, str) else json.dumps(current_input).encode('utf-8')
            output_bytes = json.dumps(current_output).encode('utf-8') if isinstance(current_output, dict) else current_output.encode('utf-8')
            self.logger.log(
                f"reverse_arbitrary_{step_name}",
                output_bytes,
                input_bytes,
                {"step_index": step_idx, "arbitrary_input": arbitrary_input}
            )
            current_output = current_input
        final_input = current_output if isinstance(current_output, str) else self.steps[0][1].reverse(current_output)
        self.logger.log(
            "reverse_arbitrary_complete",
            target_output.encode('utf-8'),
            final_input.encode('utf-8'),
            {"reconstructed_input": final_input, "arbitrary_input": arbitrary_input}
        )
        return final_input

    def save_flow_log(self, filename: str) -> None:
        with open(filename, 'w') as f:
            for entry in self.flow_log:
                f.write(json.dumps(entry) + '\n')

def define_default_flow(model_name: str = "distilgpt2", num_layers: int = 2) -> LLMFlow:
    reference_input = "Hello, world!"
    model_manager = ModelManager(model_name)
    steps = [
        ("tokenize", TokenizerModule(model_manager), {}),
        ("embed", EmbedderModule(model_manager), {}),
    ]
    for i in range(min(num_layers, 6)):
        steps.append((f"transformer_layer_{i}", TransformerLayerModule(model_manager, i), {}))
    steps.append(("decode", DecoderModule(model_manager), {"temperature": 1.0, "top_k": 50}))
    return LLMFlow(steps, reference_input)

def main():
    parser = argparse.ArgumentParser(description="LLM Flow Demonstrator with Tracing")
    parser.add_argument("--input", type=str, default="Hello, world!", help="Input text")
    parser.add_argument("--arbitrary-input", type=str, help="Arbitrary input for reverse")
    parser.add_argument("--target-output", type=str, help="Target output to reverse")
    parser.add_argument("--model-name", type=str, default="distilgpt2", help="Hugging Face model name")
    parser.add_argument("--log-file", type=str, default="llm_flow_log.jsonl", help="Log file path")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer layers")
    parser.add_argument("--commit-sampling", type=str, help="Commit sampling (seed:temperature)")
    parser.add_argument("--reveal-sampling", type=str, help="Reveal sampling (seed:temperature)")
    parser.add_argument("--user-id", type=str, default="user1", help="User ID for sampling operations")
    parser.add_argument("--fee", type=int, default=1000, help="Fee for sampling reveal")
    args = parser.parse_args()

    input_text = args.input

    engine_a = LLMEntropyEngine()
    engine_b = LLMEntropyEngine()
    engine_c = LLMEntropyEngine()
    coordinator = LLMCoordinator(engine_a, engine_b, engine_c)

    if args.commit_sampling:
        try:
            seed, temp = map(float, args.commit_sampling.split(":"))
            coordinator.commit_sampling_all(args.user_id, int(seed), temp, int(seed), temp, int(seed), temp)
            print(f"Committed sampling for user {args.user_id}: seed={seed}, temp={temp}")
        except ValueError as e:
            print(f"Sampling commit error: {str(e)}")
        return

    if args.reveal_sampling:
        try:
            seed, temp = map(float, args.reveal_sampling.split(":"))
            coordinator.reveal_sampling_all(args.user_id, int(seed), temp, int(seed), temp, int(seed), temp, args.fee)
            print(f"Revealed sampling for user {args.user_id}: seed={seed}, temp={temp}")
        except ValueError as e:
            print(f"Sampling reveal error: {str(e)}")
        return

    flow = define_default_flow(args.model_name, args.num_layers)
    try:
        result = flow.execute(input_text)
        flow.save_flow_log(args.log_file)
        print(f"Input: {input_text}")
        print(f"Output: {result}")

        reconstructed = flow.reverse(result)
        print(f"Reconstructed input: {reconstructed}")
        print(f"Reverse successful: {reconstructed == input_text}")

        if args.arbitrary_input and args.target_output:
            arbitrary_reconstructed = flow.reverse_arbitrary(args.target_output, args.arbitrary_input)
            print(f"Arbitrary input: {args.arbitrary_input}")
            print(f"Target output: {args.target_output}")
            print(f"Arbitrary reconstructed input: {arbitrary_reconstructed}")
    except ValueError as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()