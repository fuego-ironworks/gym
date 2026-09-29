import unittest

import torch
from torch import nn

from llm.models.lora_differential import RankOneLoRA


class RankOneLoRATest(unittest.TestCase):
    def make_adapter(self):
        torch.manual_seed(11)
        base = nn.Linear(3, 2, bias=False)
        base.weight.requires_grad_(False)
        return RankOneLoRA(base, seed=7)

    def test_zero_initialized_b_is_exact_noop(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        expected = adapter.base(x)
        actual = adapter(x)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(torch.count_nonzero(adapter.b).item(), 0)
        self.assertEqual(torch.count_nonzero(adapter.delta_weight()).item(), 0)

    def test_disable_and_zero_merge_are_exact(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        expected = adapter(x)
        adapter.enabled = False
        self.assertTrue(torch.equal(adapter(x), expected))
        adapter.enabled = True
        adapter.merge_()
        self.assertTrue(torch.equal(adapter(x), expected))
        adapter.unmerge_()
        self.assertTrue(torch.equal(adapter(x), expected))

    def test_first_step_moves_b_not_a(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        target = torch.tensor([[0.5, -0.25]])
        optimizer = torch.optim.SGD([adapter.a, adapter.b], lr=0.1)
        a_before = adapter.a.detach().clone()
        b_before = adapter.b.detach().clone()
        base_before = adapter.base.weight.detach().clone()

        loss = (adapter(x) - target).square().sum()
        loss.backward()
        self.assertEqual(torch.count_nonzero(adapter.a.grad).item(), 0)
        self.assertNotEqual(torch.count_nonzero(adapter.b.grad).item(), 0)
        optimizer.step()

        self.assertTrue(torch.equal(adapter.a, a_before))
        self.assertFalse(torch.equal(adapter.b, b_before))
        self.assertTrue(torch.equal(adapter.base.weight, base_before))

    def test_second_step_can_move_a(self):
        adapter = self.make_adapter()
        x = torch.tensor([[1.0, 2.0, -1.0]])
        target = torch.tensor([[0.5, -0.25]])
        optimizer = torch.optim.SGD([adapter.a, adapter.b], lr=0.1)

        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = (adapter(x) - target).square().sum()
            loss.backward()
            if step == 1:
                self.assertNotEqual(torch.count_nonzero(adapter.a.grad).item(), 0)
            optimizer.step()


if __name__ == "__main__":
    unittest.main()
