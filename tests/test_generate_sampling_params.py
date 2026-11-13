"""CPU scheduling regressions; model execution and tokenization are replaced."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import Scheduler
from nanovllm.sampling_params import SamplingParams
from serving_bench import warm_up


class GenerateSamplingParamsTests(unittest.TestCase):

    def setUp(self):
        self.engine = LLMEngine.__new__(LLMEngine)
        self.engine.scheduler = Scheduler(SimpleNamespace(
            chunked_prefill=False,
            max_model_len=1024,
            max_num_seqs=8,
            max_num_batched_tokens=512,
            eos=-1,
            num_kvcache_blocks=8,
            kvcache_block_size=256,
        ))
        self.engine.tokenizer = Mock()
        self.engine.tokenizer.encode.side_effect = lambda text: [ord(char) for char in text]
        self.engine.tokenizer.decode.side_effect = lambda ids: str(ids)
        self.engine.model_runner = Mock()
        self.engine.model_runner.call.side_effect = self.run_model

    def run_model(self, method, seqs):
        self.assertEqual(method, "run")
        return [seq.prompt_token_ids[0] + 100 for seq in seqs], list(range(len(seqs)))

    def generate(self, prompts, params):
        return self.engine.generate(prompts, params, use_tqdm=False)

    def assert_idle(self):
        self.assertTrue(self.engine.is_finished())
        self.assertTrue(all(block.ref_count == 0
                            for block in self.engine.scheduler.block_manager.blocks))

    def test_shorter_parameter_list_is_rejected_before_any_work(self):
        self.check_mismatch([[1], [2]], [SamplingParams(max_tokens=1)])

    def test_longer_parameter_list_is_rejected_before_any_work(self):
        self.check_mismatch([[1]], [SamplingParams(max_tokens=1)] * 2)

    def test_empty_parameter_list_is_rejected_for_nonempty_prompts(self):
        self.check_mismatch([[1]], [])

    def test_nonempty_parameter_list_is_rejected_for_empty_prompts(self):
        self.check_mismatch([], [SamplingParams(max_tokens=1)])

    def test_string_prompts_are_not_tokenized_on_mismatch(self):
        self.check_mismatch(["first", "second"], [SamplingParams(max_tokens=1)])

    def check_mismatch(self, prompts, params):
        with patch("nanovllm.engine.llm_engine.tqdm") as progress:
            with self.assertRaisesRegex(
                ValueError, f"Expected {len(prompts)} sampling parameters, got {len(params)}"
            ):
                self.engine.generate(prompts, params)
            progress.assert_not_called()
        self.engine.tokenizer.encode.assert_not_called()
        self.engine.tokenizer.decode.assert_not_called()
        self.engine.model_runner.call.assert_not_called()
        self.assert_idle()

    def test_rejection_preserves_pending_work_and_allows_recovery(self):
        self.engine.add_request([9], SamplingParams(max_tokens=2))
        self.engine.step()
        self.engine.add_request([8], SamplingParams(max_tokens=1))
        scheduler = self.engine.scheduler
        before = self.snapshot()
        self.engine.model_runner.call.reset_mock()
        with self.assertRaises(ValueError):
            self.generate([[1], [2]], [SamplingParams(max_tokens=1)])
        self.assertEqual(before, self.snapshot())
        self.engine.model_runner.call.assert_not_called()
        self.assertEqual([seq.prompt_token_ids for seq in scheduler.waiting], [[8]])
        outputs = self.generate([[3]], SamplingParams(max_tokens=1))
        self.assertEqual([x["token_ids"] for x in outputs], [[109, 109], [108], [103]])
        self.assert_idle()

    def snapshot(self):
        scheduler = self.engine.scheduler
        return (
            [(seq.seq_id, seq.status, list(seq.token_ids), seq.num_cached_tokens,
              seq.num_new_tokens, list(seq.block_table))
             for seq in list(scheduler.running) + list(scheduler.waiting)],
            [(block.ref_count, block.hash) for block in scheduler.block_manager.blocks],
            list(scheduler.block_manager.free_block_ids),
        )

    def test_matching_list_preserves_request_order_and_limits(self):
        outputs = self.generate([[2], [1]], [SamplingParams(max_tokens=2),
                                           SamplingParams(max_tokens=1)])
        self.assertEqual([x["token_ids"] for x in outputs], [[102, 102], [101]])
        self.assert_idle()

    def test_shared_parameters_still_broadcast(self):
        outputs = self.generate([[2], [1]], SamplingParams(max_tokens=2))
        self.assertEqual([x["token_ids"] for x in outputs], [[102, 102], [101, 101]])
        self.assert_idle()

    def test_empty_prompts_and_empty_parameter_list(self):
        self.assertEqual(self.generate([], []), [])
        self.engine.model_runner.call.assert_not_called()
        self.assert_idle()

    def test_empty_prompts_and_shared_parameters(self):
        self.assertEqual(self.generate([], SamplingParams(max_tokens=1)), [])
        self.engine.model_runner.call.assert_not_called()
        self.assert_idle()

    def test_benchmark_warmup_below_fifty_measured_requests(self):
        self.check_warmup(8)

    def test_benchmark_warmup_with_fifty_measured_requests(self):
        self.check_warmup(50)

    def test_benchmark_warmup_with_default_measured_requests(self):
        self.check_warmup(256)

    def test_benchmark_warmup_with_readme_measured_requests(self):
        self.check_warmup(1024)

    def check_warmup(self, num_requests):
        args = SimpleNamespace(num_requests=num_requests, random_input_len=4,
                               random_output_len=2)
        generated = []
        generate = self.engine.generate

        def record_generate(prompts, params):
            outputs = generate(prompts, params, use_tqdm=False)
            generated.append((prompts, params, outputs))
            return outputs

        with patch.object(self.engine, "generate", side_effect=record_generate):
            warm_up(self.engine, args)
        self.assertEqual(len(generated), 1)
        prompts, params, outputs = generated[0]
        self.assertEqual(len(prompts), 50)
        self.assertEqual(len(params), len(prompts))
        self.assertEqual(len(outputs), len(prompts))
        self.assertTrue(all(len(prompt) == args.random_input_len for prompt in prompts))
        self.assertTrue(all(sp.max_tokens == args.random_output_len and sp.ignore_eos
                            and sp.temperature == 0.6 for sp in params))
        self.assertEqual([output["token_ids"] for output in outputs],
                         [[prompt[0] + 100] * args.random_output_len for prompt in prompts])
        self.assert_idle()


if __name__ == "__main__":
    unittest.main()
