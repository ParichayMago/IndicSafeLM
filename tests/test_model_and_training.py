"""Model, objective and checkpoint checks from Tables 4.3, 6.2 and Appendix B.3."""

import math

import pytest
import torch
import torch.nn.functional as F

from indicsafelm.model.checkpoint import CheckpointError, atomic_write_dir, load_bundle, save_bundle
from indicsafelm.model.transformer import ModelConfig, ScalarHeadModel, TransformerLM, apply_rope, rope_tables
from indicsafelm.training.core import MixedPrecisionTrainer, OptimConfig, lr_at, masked_token_nll, right_pad
from indicsafelm.training.pretrain import BalancedSampler

TINY = ModelConfig(vocab_size=97, n_layer=2, d_model=32, n_head=4, d_ff=64, max_seq_len=64, dropout=0.0)


def tiny(seed=0):
    torch.manual_seed(seed)
    return TransformerLM(TINY).eval()


def test_default_config_matches_report_parameter_count():
    model = TransformerLM(ModelConfig())
    assert model.lm_head.weight is model.embed.weight  # tied embeddings
    assert 41_000_000 < model.num_parameters() < 42_500_000  # report: ~41.6M plus small terms


def test_output_shape_matches_vocabulary():
    logits = tiny()(torch.randint(0, 97, (3, 10)))
    assert logits.shape == (3, 10, TINY.vocab_size)


def test_causal_mask_has_no_future_leakage():
    model = tiny()
    x = torch.randint(0, 97, (1, 12))
    y = x.clone()
    y[0, 8:] = (y[0, 8:] + 5) % 97  # alter only future tokens
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_one_token_shift_and_loss_denominator():
    ids = torch.tensor([[5, 6, 7, 8]])
    logits = torch.full((1, 3, 97), -1e4)
    for pos, target in enumerate([6, 7, 8]):  # position t must predict token t+1
        logits[0, pos, target] = 0.0
    nll, count = masked_token_nll(logits, ids[:, 1:], torch.ones(1, 3))
    assert count.item() == 3 and nll.item() < 1e-3
    wrong, _ = masked_token_nll(logits, ids[:, :-1], torch.ones(1, 3))  # predicting itself is penalised
    assert wrong.item() > 100


def test_padding_does_not_change_per_token_loss():
    model = tiny()
    seqs = [[1, 4, 9, 16, 25], [2, 3, 5]]
    ids, mask = right_pad(seqs, pad_id=0)
    with torch.no_grad():
        nll, count = masked_token_nll(model(ids[:, :-1]), ids[:, 1:], mask[:, 1:])
        separate = sum(masked_token_nll(model(torch.tensor([s[:-1]])), torch.tensor([s[1:]]), torch.ones(1, len(s) - 1))[0]
                       for s in seqs)
    assert count.item() == 4 + 2
    assert math.isclose(nll.item(), separate.item(), rel_tol=1e-4)


def test_rope_scores_depend_only_on_relative_position():
    cos, sin = rope_tables(8, 64, 10000.0)
    q, k = torch.randn(1, 1, 1, 8), torch.randn(1, 1, 1, 8)

    def score(pq, pk):
        rq = apply_rope(q, cos[pq].view(1, 1, 1, -1), sin[pq].view(1, 1, 1, -1))
        rk = apply_rope(k, cos[pk].view(1, 1, 1, -1), sin[pk].view(1, 1, 1, -1))
        return (rq * rk).sum().item()

    assert math.isclose(score(3, 1), score(13, 11), rel_tol=1e-4, abs_tol=1e-5)


def test_left_padding_and_kv_cache_match_plain_forward():
    model = tiny()
    seq = torch.tensor([[7, 3, 11, 19, 23]])
    with torch.no_grad():
        plain = model(seq)
        padded = torch.cat([torch.zeros(1, 3, dtype=torch.long), seq], dim=1)
        mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
        left = model(padded, attention_mask=mask)
        assert torch.allclose(plain[0], left[0, 3:], atol=1e-4)
        first, past = model(seq[:, :3], use_cache=True)
        rest, _ = model(seq[:, 3:], attention_mask=torch.ones(1, 5, dtype=torch.long), past_kv=past, use_cache=True)
        assert torch.allclose(plain[0, 3:], rest[0], atol=1e-4)


def test_scalar_head_reads_last_valid_position():
    head = ScalarHeadModel(tiny()).eval()
    ids, mask = right_pad([[4, 5, 6], [4, 5, 6, 0, 0, 0]], pad_id=0)
    mask[1] = torch.tensor([1, 1, 1, 0, 0, 0])
    with torch.no_grad():
        scores = head.score(ids, mask)
    assert math.isclose(scores[0].item(), scores[1].item(), rel_tol=1e-4, abs_tol=1e-5)


def test_pairwise_reward_loss_sign():
    better = -F.logsigmoid(torch.tensor(2.0) - torch.tensor(-1.0))
    worse = -F.logsigmoid(torch.tensor(-1.0) - torch.tensor(2.0))
    assert better < math.log(2) < worse


def test_lr_schedule_warmup_then_cosine():
    cfg = OptimConfig(lr=1e-3, warmup_steps=10, total_steps=110, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(1e-4)
    assert lr_at(9, cfg) == pytest.approx(1e-3)
    assert lr_at(110, cfg) == pytest.approx(1e-4)
    assert lr_at(60, cfg) < lr_at(20, cfg)


def test_tiny_batch_overfits_with_mixed_precision_trainer():
    model = tiny(1).train()
    trainer = MixedPrecisionTrainer(model, OptimConfig(lr=3e-3, warmup_steps=1, total_steps=200, weight_decay=0.0), "fp32")
    batch = torch.randint(0, 97, (4, 16))
    first = None
    for step in range(60):
        nll, count = masked_token_nll(trainer.compute(batch[:, :-1]), batch[:, 1:], torch.ones(4, 15))
        loss = nll / count
        first = first or loss.item()
        loss.backward()
        assert trainer.step(step)["finite"]
    assert loss.item() < 0.3 * first


def test_bf16_compute_copy_tracks_master_weights():
    model = tiny(2)
    trainer = MixedPrecisionTrainer(model, OptimConfig(lr=1e-2, warmup_steps=1, total_steps=10), "bf16")
    assert trainer.compute is not trainer.master
    x = torch.randint(0, 97, (2, 8))
    trainer.compute(x).float().mean().backward()
    trainer.step(0)
    for pm, pc in zip(trainer.master_params, trainer.compute_params):
        assert pc.dtype == torch.bfloat16 and torch.allclose(pm.to(torch.bfloat16), pc)


def test_sampler_state_round_trip_is_deterministic():
    streams = {"hi": torch.arange(1000).numpy().astype("uint16"), "bn": torch.arange(1000).numpy().astype("uint16")}
    a = BalancedSampler(streams, 17, seed=3)
    a.sample(4)
    state = a.state_dict()
    expected, _ = a.sample(4)
    b = BalancedSampler(streams, 17, seed=99)
    b.load_state_dict(state)
    got, _ = b.sample(4)
    assert torch.equal(expected, got)


class FakeTok:
    fingerprint = "a" * 64
    vocab_size = 97


def test_checkpoint_round_trip_and_mismatch_errors(tmp_path):
    from indicsafelm.tokenizer import TokenizerMismatch

    model = tiny(4)
    save_bundle(tmp_path / "ck", model, role="base", model_config=TINY, tokenizer_fingerprint=FakeTok.fingerprint, run_id="r", step=1)
    loaded, meta = load_bundle(tmp_path / "ck", FakeTok())
    assert meta["role"] == "base"
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), loaded.state_dict().values()))

    class OtherTok(FakeTok):
        fingerprint = "b" * 64

    with pytest.raises(TokenizerMismatch):
        load_bundle(tmp_path / "ck", OtherTok())
    (tmp_path / "ck" / "COMPLETE").unlink()
    with pytest.raises(CheckpointError):
        load_bundle(tmp_path / "ck", FakeTok())


def test_atomic_write_leaves_no_partial_bundle(tmp_path):
    def broken(tmp):
        (tmp / "weights.pt").write_text("partial")
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        atomic_write_dir(tmp_path / "ck", broken)
    assert not (tmp_path / "ck").exists()
    assert not any(tmp_path.iterdir())
