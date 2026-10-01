"""Portable inference-only hooks from the historical 400-example controls.

These implement the 2026-09-15 protocol. They are not a claimed reconstruction
of the later author-supplied full-benchmark experiment.
The new full_eval.py explicitly applies these same documented hooks to complete
datasets, preserving the distinction between a new run and historical aggregates.
"""
import torch


class VisualStateControl:
    MODES = ("normal", "blocked", "frozen")

    def __init__(self, recurrent, mode="normal"):
        if mode not in self.MODES:
            raise ValueError(mode)
        if (recurrent.config.H_cycles, recurrent.config.L_cycles) != (2, 3):
            raise ValueError("The released historical control requires H2L3")
        if len(recurrent.L_module.layers) != 16 or len(recurrent.H_module.layers) != 16:
            raise ValueError("Expected 16 layers in each shared module")
        self.recurrent, self.mode = recurrent, mode
        self.handles, self.audit = [], []

    def __enter__(self):
        self.handles.append(self.recurrent.register_forward_pre_hook(self._begin, with_kwargs=True))
        for kind, stack in (("L", self.recurrent.L_module), ("H", self.recurrent.H_module)):
            self.handles.append(stack.register_forward_pre_hook(self._stack_pre(kind), with_kwargs=True))
            self.handles.extend(layer.register_forward_hook(self._layer_post(kind)) for layer in stack.layers)
            self.handles.append(stack.register_forward_hook(self._stack_post(kind)))
        self.handles.append(self.recurrent.register_forward_hook(self._end))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _begin(self, module, args, kwargs):
        mask = kwargs.get("visual_mask")
        if mask is None or mask.ndim != 2 or mask.shape[0] != 1 or not mask.any():
            raise ValueError("Batch size 1 and a nonempty keyword visual_mask are required")
        self.positions = mask[0].nonzero(as_tuple=False).flatten()
        self.l_calls = self.h_calls = self.layer_restores = self.stack_restores = self.carry_restores = 0
        self.saved_l = self.reference = None
        self.active_kind = None
        self.active_cycle2 = False

    def _replace(self, states, visual):
        edited = states.clone()
        edited[:, self.positions, :] = visual
        return edited

    def _stack_pre(self, kind):
        def hook(module, args, kwargs):
            if not args or not isinstance(args[0], torch.Tensor):
                raise TypeError("The checkpoint stack must receive positional hidden states")
            if kind == "L":
                call = self.l_calls
                self.l_calls += 1
                self.active_cycle2 = call >= 3
                if call == 3:
                    # Actual L1 stack input, after the existing re-grounding and
                    # the normal L-state/H-state combination.
                    self.reference = args[0].index_select(1, self.positions).clone()
            else:
                call = self.h_calls
                self.h_calls += 1
                self.active_cycle2 = call == 1
            self.active_kind = kind
            if self.mode == "frozen" and self.active_cycle2:
                assert self.reference is not None
                return (self._replace(args[0], self.reference),) + args[1:], kwargs
        return hook

    def _layer_post(self, kind):
        def hook(module, args, output):
            if self.mode == "frozen" and self.active_cycle2 and self.active_kind == kind:
                self.layer_restores += 1
                return self._replace(output, self.reference)
        return hook

    def _stack_post(self, kind):
        def hook(module, args, output):
            edited = None
            if kind == "L" and self.l_calls == 3:
                # Persistent L-state at Cycle-1 L3 exit. H1 and re-grounding
                # subsequently update H-state, leaving this L-state intact.
                self.saved_l = output.index_select(1, self.positions).clone()
            if self.mode == "blocked" and kind == "L" and self.l_calls >= 4:
                assert self.saved_l is not None
                edited = self._replace(output, self.saved_l)
                self.carry_restores += 1
            elif self.mode == "frozen" and self.active_cycle2:
                # The stack's final RMSNorm also needs a restore.
                edited = self._replace(output, self.reference)
                self.stack_restores += 1
            self.active_kind = None
            self.active_cycle2 = False
            return edited
        return hook

    def _end(self, module, args, output):
        assert self.l_calls == 6 and self.h_calls == 2
        if self.mode == "blocked":
            assert self.carry_restores == 3
        elif self.mode == "frozen":
            assert self.layer_restores == 64 and self.stack_restores == 4
        self.audit.append({"mode": self.mode, "L_calls": self.l_calls, "H_calls": self.h_calls,
                           "carry_restores": self.carry_restores, "layer_restores": self.layer_restores,
                           "stack_restores": self.stack_restores})
