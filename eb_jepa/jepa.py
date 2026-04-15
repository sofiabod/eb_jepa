import torch
import torch.nn as nn

from eb_jepa.utils.logging import get_logger

logging = get_logger(__name__)


class JEPAbase(nn.Module):
    """Base JEPA class for planning and inference only. Use JEPA subclass for training."""

    def __init__(self, encoder, aencoder, predictor):
        """Initialize JEPAbase with encoder, action encoder, and predictor."""
        super().__init__()
        # Observation Encoder
        self.encoder = encoder
        # Action Encoder
        self.action_encoder = aencoder
        # Predictor
        self.predictor = predictor
        self.single_unroll = getattr(self.predictor, "is_rnn", False)

    @torch.no_grad()
    def encode(self, observations):
        """Encode a sequence of observations and return the encoder output."""
        return self.encoder(observations)


class JEPA(JEPAbase):
    """Trainable JEPA with prediction loss and anti-collapse regularizer."""

    def __init__(
        self,
        encoder,
        aencoder,
        predictor,
        regularizer,
        predcost,
    ):
        """Initialize JEPA with regularizer and prediction cost in addition to base components.

        Args:
            encoder: Observation encoder network.
            aencoder: Action encoder network.
            predictor: State predictor network.
            regularizer: Anti-collapse regularizer.
            predcost: Prediction loss function.
        """
        super().__init__(encoder, aencoder, predictor)
        self.regularizer = regularizer
        self.predcost = predcost

    def forward(self, *args, **kwargs):
        """Route through ``unroll()`` so DDP gradient synchronization works.

        DDP only triggers all-reduce hooks when ``forward()`` is called.
        Calling ``unroll()`` directly on a DDP-wrapped model would bypass
        gradient synchronization.  Use ``model(obs, actions, ...)`` in
        training loops instead of ``model.unroll(...)``.
        """
        return self.unroll(*args, **kwargs)

    @torch.no_grad()
    def infer(self, observations, actions):
        """Produce single-step predictions over all sequence elements in parallel."""
        preds, _, _ = self.unroll(
            observations,
            actions,
            nsteps=1,
            unroll_mode="parallel",
            compute_loss=False,
            return_all_steps=True,
        )
        return preds[0]

    def unroll(
        self,
        observations,
        actions,
        nsteps=1,
        unroll_mode="parallel",
        ctxt_window_time=1,
        compute_loss=True,
        return_all_steps=False,
        _precomputed_state=None,
        _precomputed_actions=None,
        stop_gradient=False,
        detach_pred_target=False,
        **kwargs,
    ):
        """Unified multi-step prediction with optional loss computation.

        This function supports both training (with loss computation) and planning/inference
        (without loss, just state prediction).

        Usage examples:
        - Training video_jepa: unroll(x, None, nsteps, unroll_mode="parallel", compute_loss=True)
        - Training ac_video_jepa with RNN: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=1, compute_loss=True)
        - Planning with ac_video_jepa: unroll(x, a, nsteps, unroll_mode="autoregressive",
          ctxt_window_time=k, compute_loss=False)
        - Inference like infern(): unroll(x, a, nsteps, unroll_mode="parallel",
          compute_loss=False, return_all_steps=True)

        Predictor behavior:
        - unroll_mode="parallel" (Conv predictor, is_rnn=False):
          Processes all timesteps in parallel. Uses predictor.context_length to
          determine how many ground truth frames to re-feed at each iteration.
          Output: [B, D, T, H', W'] (same length as input, predictions replace non-context).
          Best for training with full ground truth trajectory available.

        - unroll_mode="autoregressive":
          Step-by-step prediction with sliding window of ctxt_window_time states.
          Each step: takes last ctxt_window_time states, predicts next, appends to sequence.
          Output: [B, D, T_context + nsteps, H', W'] (context + predictions appended).
          Best for planning/inference where future ground truth is not available.
          Note: RNN predictors (is_rnn=True) are a special case with ctxt_window_time=1.

        Args:
            observations: [B, C, T, H, W] - observation sequence
                For training (compute_loss=True): full trajectory with ground truth
                For planning (compute_loss=False): context frames only
            actions: [B, A, T_actions] - action sequence, or None for state-only prediction
                T_actions >= nsteps + ctxt_window_time required for autoregressive mode
            nsteps: number of prediction steps
            unroll_mode: "parallel" or "autoregressive"
                - "parallel": Process all timesteps, refeed GT context on left
                - "autoregressive": Step-by-step, append predictions on right
            ctxt_window_time: Context window size for autoregressive mode.
                For RNN predictors (is_rnn=True), this is effectively 1.
            compute_loss: Whether to compute losses (requires ground truth observations)
            return_all_steps: If True, return list of predictions at each step (like infern).
                If False, return only the final predicted states.
            _precomputed_state: Optional pre-encoded state, skips encoder when provided.
                Used by HierarchicalJEPA to pass states already encoded at this level.
            _precomputed_actions: Optional pre-encoded actions, skips action encoder
                when provided. Used by HierarchicalJEPA for pre-aggregated actions.
            stop_gradient: If True, detach predicted context states before re-feeding
                to the predictor on iterations after the first. The first predictor
                call (teacher-forcing on encoder outputs) always has full gradients;
                detaching only applies to subsequent iterations where the predictor's
                own outputs are re-fed as inputs.
            detach_pred_target: If True, detach encoder states used as prediction
                targets. Prevents prediction-loss gradients from reaching the encoder,
                so only the regularizer (SIGReg + IDM) shapes encoder representations.

        Returns:
            Tuple of (predicted_states, losses) where:
            - If return_all_steps=False:
              predicted_states: [B, D, T_out, H', W'] - final predicted state sequence
            - If return_all_steps=True:
              predicted_states: List[Tensor] of length nsteps, each [B, D, T_out, H', W']
            - encoded_states: [B, D, T, H', W'] - ground truth encoded observations
            - losses: None if compute_loss=False, otherwise tuple of 5 elements:
              (total_loss, reg_loss, reg_loss_unweighted, reg_loss_dict, pred_loss)
        """
        state = (
            _precomputed_state
            if _precomputed_state is not None
            else self.encoder(observations)
        )

        pred_target = state.detach() if detach_pred_target else state
        context_length = getattr(self.predictor, "context_length", 1)

        # Compute regularization loss if needed
        if compute_loss:
            rloss, rloss_unweight, rloss_dict = self.regularizer(state, actions)
            ploss = torch.tensor(0.0, device=state.device)
        else:
            rloss = rloss_unweight = rloss_dict = ploss = None

        # Encode actions
        if _precomputed_actions is not None:
            actions_encoded = _precomputed_actions
        elif actions is not None:
            actions_encoded = self.action_encoder(actions)
        else:
            actions_encoded = None

        # Collect all steps if requested
        all_steps = [] if return_all_steps else None

        # Parallel mode: process all timesteps at once, refeed GT context
        if unroll_mode == "parallel":
            predicted_states = state
            for step_i in range(nsteps):
                pred_input = predicted_states
                if stop_gradient and step_i > 0:
                    pred_input = torch.cat(
                        [
                            state[:, :, :context_length],
                            predicted_states[:, :, context_length:].detach(),
                        ],
                        dim=2,
                    )
                if actions_encoded is not None:
                    T_a = actions_encoded.size(2)
                    predicted_states = self.predictor(
                        pred_input[:, :, :T_a], actions_encoded
                    )
                else:
                    predicted_states = self.predictor(pred_input, actions_encoded)[
                        :, :, :-1
                    ]
                if return_all_steps:
                    all_steps.append(predicted_states)
                predicted_states = torch.cat(
                    (state[:, :, :context_length], predicted_states), dim=2
                )
                if compute_loss:
                    ploss += self.predcost(pred_target, predicted_states) / nsteps

        # Autoregressive mode: step-by-step with sliding window
        #
        # With W = effective_ctxt_window and N = nsteps:
        #
        #   GT states:  s_0  s_1  ...  s_{W-1}  s_W  s_{W+1} ... s_{W+N-1}
        #               |--- context ---|        |--- predictions ---------|
        #   Actions:    a_0  a_1  ...  a_{W-1}  a_W  a_{W+1} ... a_{W+N-2}
        #
        #   Step i=0: context = [s_0..s_{W-1}], actions = [a_0..a_{W-1}] -> predict s_W
        #   Step i=1: context = [s_1..s_W],     actions = [a_1..a_W]     -> predict s_{W+1}
        #   ...
        #   Step i=k: context = predicted_states[:, :, k:k+W], actions = actions[:, :, k:k+W]
        #
        # Requires: nsteps + W - 1 <= len(actions)
        # Note: RNN predictors (is_rnn=True) are a special case with W=1.
        elif unroll_mode == "autoregressive":
            effective_ctxt_window = 1 if self.single_unroll else ctxt_window_time
            if (
                actions is not None
                and nsteps + effective_ctxt_window - 1 > actions.size(2)
            ):
                raise ValueError(
                    f"nsteps ({nsteps}) + ctxt_window ({effective_ctxt_window}) "
                    f"= {nsteps + effective_ctxt_window} exceeds action sequence "
                    f"length ({actions.size(2)}). Increase num_frames or reduce "
                    f"nsteps/ctxt_window_time."
                )

            W = effective_ctxt_window
            predicted_states = state[:, :, :W]  # [B, D, W, H', W']
            for i in range(nsteps):
                context_states = predicted_states[:, :, -W:]  # [B, D, W, H', W']
                if stop_gradient and i > 0:
                    context_states = context_states.detach()
                if actions_encoded is not None:
                    buf_len = predicted_states.size(2)
                    act_start = max(0, buf_len - W)
                    context_actions = actions_encoded[
                        :, :, act_start:buf_len
                    ]  # [B, A, ≤W]
                else:
                    context_actions = None
                pred_step = self.predictor(context_states, context_actions)[
                    :, :, -1:
                ]  # [B, D, 1, H', W']
                predicted_states = torch.cat(
                    [predicted_states, pred_step], dim=2
                )  # [B, D, W+i+1, H', W']
                if return_all_steps:
                    all_steps.append(predicted_states.clone())
                if compute_loss:
                    target = pred_target[:, :, W + i : W + i + 1]  # [B, D, 1, H', W']
                    ploss += self.predcost(pred_step, target) / nsteps
        else:
            raise ValueError(f"Unknown unroll_mode: {unroll_mode}")

        # Compute total loss and return
        if compute_loss:
            loss = rloss + ploss
            losses = (loss, rloss, rloss_unweight, rloss_dict, ploss)
        else:
            losses = None

        # Return all steps or just final state
        if return_all_steps:
            return all_steps, state, losses
        else:
            return predicted_states, state, losses


class JEPAProbe(nn.Module):
    """JEPA with a trainable prediction head. The JEPA encoder is kept fixed."""

    def __init__(self, jepa, head, hcost):
        """Initialize with a frozen JEPA, prediction head, and head loss function."""
        super().__init__()
        self.jepa = jepa
        self.head = head
        self.hcost = hcost

    def _raw_head(self):
        """Return the unwrapped head (strips DDP/compile wrappers)."""
        from eb_jepa.utils.distributed import unwrap_model

        return unwrap_model(self.head)

    @torch.no_grad()
    def infer(self, observations):
        """Encode observations through JEPA and apply the prediction head."""
        state = self.jepa.encode(observations)
        return self._raw_head()(state)

    @torch.no_grad()
    def apply_head(self, embeddings):
        """Decode embeddings using the head and denormalize predictions.

        Returns predictions in the original (unnormalized) target space.
        Uses the unwrapped head to avoid DDP collective ops (buffer broadcasts)
        that would desync NCCL when called on rank 0 only during eval.
        """
        raw = self._raw_head()
        pred = raw(embeddings)
        if hasattr(raw, "denormalize"):
            pred = raw.denormalize(pred)
        return pred

    def forward(self, observations, targets):
        """Forward pass for training the head (JEPA encoder gradients are detached).

        Targets are Z-score normalized before computing the loss when the head
        has position statistics registered.
        """
        with torch.no_grad():
            state = self.jepa.encode(observations)
        output = self.head(state.detach())
        if hasattr(self.head, "normalize_targets"):
            targets = self.head.normalize_targets(targets)
        return self.hcost(output, targets)


class JEPAWithCostModule(JEPA):
    """JEPA with an optional cost module for planning objectives.

    The cost_module is a trainable network (e.g., a projector) that can be used
    to define custom planning cost functions, such as computing distances in a
    learned projected space.
    """

    def __init__(
        self,
        encoder,
        aencoder,
        predictor,
        regularizer,
        predcost,
        cost_module=None,
    ):
        """Initialize JEPA with an optional cost module for planning.

        Args:
            encoder: Observation encoder network.
            aencoder: Action encoder network.
            predictor: State predictor network.
            regularizer: Anti-collapse regularizer.
            predcost: Prediction loss function.
            cost_module: Optional trainable module for planning cost (e.g., Projector).
        """
        super().__init__(
            encoder,
            aencoder,
            predictor,
            regularizer,
            predcost,
        )
        self.cost_module = cost_module

    def unroll(self, observations, actions, **kwargs):
        predicted_states, enc_states, losses = super().unroll(
            observations, actions, **kwargs
        )

        if losses is None or self.cost_module is None:
            return predicted_states, enc_states, losses

        cost_loss, cost_loss_dict = self.cost_module(enc_states)

        total_loss, rloss, rloss_unweight, rloss_dict, ploss = losses
        rloss_dict.update(cost_loss_dict)
        return (
            predicted_states,
            enc_states,
            (total_loss + cost_loss, rloss, rloss_unweight, rloss_dict, ploss),
        )
