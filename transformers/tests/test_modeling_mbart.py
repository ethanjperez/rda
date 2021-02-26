import unittest

from transformers import is_torch_available
from transformers.file_utils import cached_property
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device

from .test_modeling_bart import TOLERANCE, _long_tensor, assert_tensors_close


if is_torch_available():
    import torch

    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        BatchEncoding,
        MBartConfig,
        MBartForConditionalGeneration,
    )


EN_CODE = 250004
RO_CODE = 250020


@require_torch
@require_sentencepiece
@require_tokenizers
class AbstractSeq2SeqIntegrationTest(unittest.TestCase):
    maxDiff = 1000  # longer string compare tracebacks
    checkpoint_name = None

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.checkpoint_name)
        return cls

    @cached_property
    def model(self):
        """Only load the model if needed."""
        model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name).to(torch_device)
        if "cuda" in torch_device:
            model = model.half()
        return model


@require_torch
@require_sentencepiece
@require_tokenizers
class MBartEnroIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "facebook/mbart-large-en-ro"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        """ Secretary-General Ban Ki-moon says his response to Russia's stepped up military support for Syria is that "there is no military solution" to the nearly five-year conflict and more weapons will only worsen the violence and misery for millions of people.""",
    ]
    tgt_text = [
        "Şeful ONU declară că nu există o soluţie militară în Siria",
        'Secretarul General Ban Ki-moon declară că răspunsul său la intensificarea sprijinului militar al Rusiei pentru Siria este că "nu există o soluţie militară" la conflictul de aproape cinci ani şi că noi arme nu vor face decât să înrăutăţească violenţa şi mizeria pentru milioane de oameni.',
    ]
    expected_src_tokens = [8274, 127873, 25916, 7, 8622, 2071, 438, 67485, 53, 187895, 23, 51712, 2, EN_CODE]

    @slow
    @unittest.skip("This has been failing since June 20th at least.")
    def test_enro_forward(self):
        model = self.model
        net_input = {
            "input_ids": _long_tensor(
                [
                    [3493, 3060, 621, 104064, 1810, 100, 142, 566, 13158, 6889, 5, 2, 250004],
                    [64511, 7, 765, 2837, 45188, 297, 4049, 237, 10, 122122, 5, 2, 250004],
                ]
            ),
            "decoder_input_ids": _long_tensor(
                [
                    [250020, 31952, 144, 9019, 242307, 21980, 55749, 11, 5, 2, 1, 1],
                    [250020, 884, 9019, 96, 9, 916, 86792, 36, 18743, 15596, 5, 2],
                ]
            ),
        }
        net_input["attention_mask"] = net_input["input_ids"].ne(1)
        with torch.no_grad():
            logits, *other_stuff = model(**net_input)

        expected_slice = torch.tensor([9.0078, 10.1113, 14.4787], device=logits.device, dtype=logits.dtype)
        result_slice = logits[0, 0, :3]
        assert_tensors_close(expected_slice, result_slice, atol=TOLERANCE)

    @slow
    def test_enro_generate_one(self):
        batch: BatchEncoding = self.tokenizer.prepare_seq2seq_batch(
            ["UN Chief Says There Is No Military Solution in Syria"]
        ).to(torch_device)
        translated_tokens = self.model.generate(**batch)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])
        # self.assertEqual(self.tgt_text[1], decoded[1])

    @slow
    def test_enro_generate(self):
        batch: BatchEncoding = self.tokenizer.prepare_seq2seq_batch(self.src_text).to(torch_device)
        translated_tokens = self.model.generate(**batch)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        assert self.tgt_text == decoded

    def test_mbart_enro_config(self):
        mbart_models = ["facebook/mbart-large-en-ro"]
        expected = {"scale_embedding": True, "output_past": True}
        for name in mbart_models:
            config = MBartConfig.from_pretrained(name)
            self.assertTrue(config.is_valid_mbart())
            for k, v in expected.items():
                try:
                    self.assertEqual(v, getattr(config, k))
                except AssertionError as e:
                    e.args += (name, k)
                    raise

    def test_mbart_fast_forward(self):
        config = MBartConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            add_final_layer_norm=True,
            return_dict=True,
        )
        lm_model = MBartForConditionalGeneration(config).to(torch_device)
        context = torch.Tensor([[71, 82, 18, 33, 46, 91, 2], [68, 34, 26, 58, 30, 2, 1]]).long().to(torch_device)
        summary = torch.Tensor([[82, 71, 82, 18, 2], [58, 68, 2, 1, 1]]).long().to(torch_device)
        result = lm_model(input_ids=context, decoder_input_ids=summary, labels=summary)
        expected_shape = (*summary.shape, config.vocab_size)
        self.assertEqual(result.logits.shape, expected_shape)


@require_torch
@require_sentencepiece
@require_tokenizers
class MBartCC25IntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "facebook/mbart-large-cc25"
    src_text = [
        " UN Chief Says There Is No Military Solution in Syria",
        " I ate lunch twice yesterday",
    ]
    tgt_text = ["Şeful ONU declară că nu există o soluţie militară în Siria", "to be padded"]

    @unittest.skip("This test is broken, still generates english")
    def test_cc25_generate(self):
        inputs = self.tokenizer.prepare_seq2seq_batch([self.src_text[0]]).to(torch_device)
        translated_tokens = self.model.generate(
            input_ids=inputs["input_ids"].to(torch_device),
            decoder_start_token_id=self.tokenizer.lang_code_to_id["ro_RO"],
        )
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text[0], decoded[0])

    @slow
    def test_fill_mask(self):
        inputs = self.tokenizer.prepare_seq2seq_batch(["One of the best <mask> I ever read!"]).to(torch_device)
        outputs = self.model.generate(
            inputs["input_ids"], decoder_start_token_id=self.tokenizer.lang_code_to_id["en_XX"], num_beams=1
        )
        prediction: str = self.tokenizer.batch_decode(
            outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )[0]
        self.assertEqual(prediction, "of the best books I ever read!")
